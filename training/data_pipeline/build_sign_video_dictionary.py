"""
Build a token -> sign clip dictionary manifest for concatenative synthesis.

Input layouts supported:
  1) <clips_root>/<TOKEN>/*.mp4
  2) <clips_root>/*.mp4 (token inferred from file stem prefix)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _normalize_token(token: str) -> str:
    tok = (token or "").strip().upper()
    tok = "_".join(tok.split())
    return tok


def _token_from_file(root: Path, video_path: Path) -> str:
    rel_parent = video_path.parent.relative_to(root)
    if str(rel_parent) not in {".", ""}:
        return _normalize_token(video_path.parent.name)
    stem = video_path.stem
    parts = stem.replace("-", "_").split("_")
    if parts:
        return _normalize_token(parts[0])
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-root", default=os.path.join("datasets", "sign_dictionary", "clips"))
    parser.add_argument("--output-manifest", default=os.path.join("datasets", "sign_dictionary", "manifest.csv"))
    parser.add_argument("--max-per-token", type=int, default=10)
    parser.add_argument("--dataset", default="local")
    parser.add_argument("--default-score", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(args.clips_root)
    if not root.exists():
        raise FileNotFoundError(f"clips root not found: {root}")

    videos = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    videos.sort()

    bucket = {}
    for vid in videos:
        token = _token_from_file(root, vid)
        if not token:
            continue
        bucket.setdefault(token, []).append(vid)

    rows: List[dict] = []
    for token in sorted(bucket.keys()):
        for vid in bucket[token][: max(1, int(args.max_per_token))]:
            rows.append(
                {
                    "token": token,
                    "video_path": str(vid).replace("\\", "/"),
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "score": float(args.default_score),
                    "dataset": args.dataset,
                }
            )

    out_path = Path(args.output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token", "video_path", "start_sec", "end_sec", "score", "dataset"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dictionary tokens: {len(bucket)}")
    print(f"Dictionary clips: {len(rows)}")
    print(f"Manifest: {out_path}")


if __name__ == "__main__":
    main()
