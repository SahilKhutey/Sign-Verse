"""
Concatenative sign video synthesis.

Inspired by modular sign-language-translator style systems:
text/gloss -> sequence of sign clips -> synthetic sentence video.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from nlp_translation.sign_grammar_converter import SignGrammarConverter


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _normalize_token(token: str) -> str:
    tok = (token or "").strip().upper()
    for ch in [",", ".", "!", "?", ";", ":", "\"", "'"]:
        tok = tok.replace(ch, "")
    tok = "_".join(tok.split())
    return tok


@dataclass
class ClipEntry:
    token: str
    video_path: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    score: float = 1.0
    dataset: str = ""


class ConcatenativeSynthesis:
    """
    Build text/gloss to sign clip plans and optionally render concatenated videos.
    """

    def __init__(
        self,
        dictionary_manifest: str = os.path.join("datasets", "sign_dictionary", "manifest.csv"),
        strict_exists: bool = False,
    ):
        self.dictionary_manifest = dictionary_manifest
        self.grammar = SignGrammarConverter()
        self.index: Dict[str, List[ClipEntry]] = {}
        self._loaded = False
        self.strict_exists = strict_exists

    def _load_manifest(self):
        if self._loaded:
            return

        if not os.path.exists(self.dictionary_manifest):
            if self.strict_exists:
                raise FileNotFoundError(
                    f"Sign dictionary manifest not found: {self.dictionary_manifest}"
                )
            self._loaded = True
            return

        if self.dictionary_manifest.lower().endswith(".json"):
            self._load_json_manifest(self.dictionary_manifest)
        else:
            self._load_csv_manifest(self.dictionary_manifest)

        for token, entries in self.index.items():
            entries.sort(key=lambda e: float(e.score), reverse=True)
            self.index[token] = entries
        self._loaded = True

    def _load_csv_manifest(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                token = _normalize_token(row.get("token", ""))
                video_path = (row.get("video_path") or "").strip()
                if not token or not video_path:
                    continue
                entry = ClipEntry(
                    token=token,
                    video_path=video_path,
                    start_sec=float(row.get("start_sec") or 0.0),
                    end_sec=float(row.get("end_sec") or 0.0),
                    score=float(row.get("score") or 1.0),
                    dataset=(row.get("dataset") or "").strip(),
                )
                self.index.setdefault(token, []).append(entry)

    def _load_json_manifest(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("entries", [])
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            token = _normalize_token(str(row.get("token", "")))
            video_path = str(row.get("video_path", "")).strip()
            if not token or not video_path:
                continue
            entry = ClipEntry(
                token=token,
                video_path=video_path,
                start_sec=float(row.get("start_sec") or 0.0),
                end_sec=float(row.get("end_sec") or 0.0),
                score=float(row.get("score") or 1.0),
                dataset=str(row.get("dataset") or "").strip(),
            )
            self.index.setdefault(token, []).append(entry)

    def text_to_gloss_tokens(self, text: str) -> List[str]:
        """
        Convert free text to gloss-like tokens (rule-based baseline).
        """
        raw = self.grammar.convert(text or "")
        return [_normalize_token(tok) for tok in raw if _normalize_token(tok)]

    def build_plan(self, gloss_tokens: Iterable[str]) -> Dict[str, object]:
        """
        Build a clip plan for provided gloss tokens.
        """
        self._load_manifest()

        plan = []
        missing = []
        found = 0
        total = 0

        for token in gloss_tokens:
            norm = _normalize_token(token)
            if not norm:
                continue
            total += 1
            candidates = self.index.get(norm, [])
            if not candidates:
                missing.append(norm)
                plan.append(
                    {
                        "token": norm,
                        "found": False,
                        "video_path": None,
                        "start_sec": 0.0,
                        "end_sec": 0.0,
                        "score": 0.0,
                        "dataset": None,
                    }
                )
                continue

            top = candidates[0]
            found += 1
            plan.append(
                {
                    "token": norm,
                    "found": True,
                    "video_path": top.video_path,
                    "start_sec": float(top.start_sec),
                    "end_sec": float(top.end_sec),
                    "score": float(top.score),
                    "dataset": top.dataset or None,
                    "alternatives": len(candidates),
                }
            )

        coverage = float(found / total) if total > 0 else 0.0
        return {
            "tokens": [p["token"] for p in plan],
            "plan": plan,
            "missing_tokens": missing,
            "total_tokens": int(total),
            "found_tokens": int(found),
            "coverage": coverage,
        }

    def plan_from_text(self, text: str) -> Dict[str, object]:
        gloss = self.text_to_gloss_tokens(text)
        out = self.build_plan(gloss)
        out["text"] = text
        out["gloss"] = gloss
        return out

    def synthesize_plan(
        self,
        plan: List[Dict[str, object]],
        output_path: str,
        fps: float = 25.0,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, object]:
        """
        Render a concatenated sentence video from a clip plan.
        """
        if not plan:
            raise ValueError("Empty plan")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(1.0, fps)),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

        written_frames = 0
        rendered_tokens = 0
        used_paths = []
        errors = []
        try:
            for item in plan:
                if not bool(item.get("found")):
                    continue
                video_path = str(item.get("video_path") or "")
                if not video_path:
                    continue
                if not os.path.exists(video_path):
                    errors.append(f"missing_clip:{video_path}")
                    continue
                ok, n = self._write_clip_segment(
                    video_path=video_path,
                    writer=writer,
                    width=width,
                    height=height,
                    start_sec=float(item.get("start_sec") or 0.0),
                    end_sec=float(item.get("end_sec") or 0.0),
                )
                if ok:
                    rendered_tokens += 1
                    used_paths.append(video_path)
                    written_frames += int(n)
                else:
                    errors.append(f"decode_failed:{video_path}")
        finally:
            writer.release()

        return {
            "output_path": output_path,
            "frames_written": int(written_frames),
            "rendered_tokens": int(rendered_tokens),
            "used_clips": used_paths,
            "errors": errors,
        }

    def synthesize_text(
        self,
        text: str,
        output_path: str,
        fps: float = 25.0,
        width: int = 640,
        height: int = 480,
    ) -> Dict[str, object]:
        """
        text -> gloss tokens -> clip plan -> rendered video.
        """
        planned = self.plan_from_text(text)
        render = self.synthesize_plan(
            plan=planned["plan"],
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
        )
        return {
            "text": text,
            "gloss": planned["gloss"],
            "coverage": planned["coverage"],
            "missing_tokens": planned["missing_tokens"],
            **render,
        }

    @staticmethod
    def _write_clip_segment(
        video_path: str,
        writer,
        width: int,
        height: int,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
    ) -> Tuple[bool, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        start_idx = max(0, int(float(start_sec) * float(src_fps)))
        if end_sec and end_sec > 0:
            end_idx = int(float(end_sec) * float(src_fps))
        else:
            end_idx = max(0, total_frames - 1) if total_frames > 0 else 10**9
        if end_idx < start_idx:
            end_idx = start_idx

        idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx < start_idx:
                idx += 1
                continue
            if idx > end_idx:
                break
            if frame is None:
                idx += 1
                continue
            resized = cv2.resize(frame, (int(width), int(height)))
            writer.write(resized)
            written += 1
            idx += 1

        cap.release()
        return written > 0, written
