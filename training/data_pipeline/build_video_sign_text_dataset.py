"""
Build video-frame-feature -> text training dataset for sign->text seq2seq.

Input CSV defaults to synthetic sentence videos:
  training-data/synthetic_sign_video_pairs.csv

Output:
  - training-data/video_keypoints/*.npy      (T, 225) frame features
  - training-data/video_sign_text_labels.csv
  - reports/video_sign_text_dataset_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from nlp_translation.sign_grammar_converter import SignGrammarConverter
from vision_pipeline.video_embedding import VideoEmbeddingExtractor


def _norm_label(gloss: str) -> str:
    g = " ".join((gloss or "").strip().upper().split())
    return g.replace(" ", "_")


def _split_by_hash(
    key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    total = max(1.0, train_ratio + val_ratio + test_ratio)
    train_cut = train_ratio / total
    val_cut = (train_ratio + val_ratio) / total
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / float(0xFFFFFFFF)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        default=os.path.join("training-data", "synthetic_sign_video_pairs.csv"),
    )
    parser.add_argument("--video-column", default="video_path")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--gloss-column", default="gloss")
    parser.add_argument("--coverage-column", default="coverage")
    parser.add_argument("--min-coverage", type=float, default=0.6)
    parser.add_argument("--keypoint-dir", default=os.path.join("training-data", "video_keypoints"))
    parser.add_argument("--labels-csv", default=os.path.join("training-data", "video_sign_text_labels.csv"))
    parser.add_argument("--label-map", default=os.path.join("training-data", "video_sign_text_label_map.json"))
    parser.add_argument("--report", default=os.path.join("reports", "video_sign_text_dataset_report.json"))
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.exists(args.input_manifest):
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")

    os.makedirs(args.keypoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.label_map) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    extractor = VideoEmbeddingExtractor(
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        allow_mock=args.allow_mock,
    )
    grammar = SignGrammarConverter()

    label_to_id: Dict[str, int] = {}
    rows_out: List[dict] = []

    total_rows = 0
    usable_rows = 0
    failed = 0
    skipped_coverage = 0

    with open(args.input_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            total_rows += 1
            if args.max_samples and usable_rows >= int(args.max_samples):
                break

            try:
                coverage = float(row.get(args.coverage_column) or 1.0)
            except Exception:
                coverage = 1.0
            if coverage < float(args.min_coverage):
                skipped_coverage += 1
                continue

            video_path = (row.get(args.video_column) or "").strip()
            text = " ".join((row.get(args.text_column) or "").strip().split())
            gloss = " ".join((row.get(args.gloss_column) or "").strip().split()).upper()
            if not text and gloss:
                text = " ".join(tok.capitalize() for tok in gloss.split())
            if not gloss and text:
                gloss = " ".join(grammar.convert(text))

            if not video_path or not os.path.exists(video_path):
                failed += 1
                continue
            if not text:
                failed += 1
                continue

            try:
                frame_feats = extractor.extract_frame_features(video_path)
                if frame_feats.ndim != 2 or frame_feats.shape[0] == 0:
                    failed += 1
                    continue
            except Exception:
                failed += 1
                continue

            filename = f"video_sample_{usable_rows:06d}.npy"
            out_path = os.path.join(args.keypoint_dir, filename)
            np.save(out_path, frame_feats.astype(np.float32))

            label_name = _norm_label(gloss or text)
            if label_name not in label_to_id:
                label_to_id[label_name] = len(label_to_id)
            label_id = int(label_to_id[label_name])

            split = _split_by_hash(
                key=f"{text}||{gloss}||{video_path}",
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )

            rows_out.append(
                {
                    "filename": filename,
                    "label_id": label_id,
                    "label_name": label_name,
                    "dataset": "synthetic_video_sentence",
                    "split": split,
                    "text": text,
                    "sentence": text,
                    "gloss": gloss,
                    "video_path": video_path,
                    "num_frames": int(frame_feats.shape[0]),
                    "feature_dim": int(frame_feats.shape[1]),
                }
            )
            usable_rows += 1

    with open(args.labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "label_id",
                "label_name",
                "dataset",
                "split",
                "text",
                "sentence",
                "gloss",
                "video_path",
                "num_frames",
                "feature_dim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    with open(args.label_map, "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2)

    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for r in rows_out:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": args.input_manifest,
        "total_rows": total_rows,
        "usable_rows": usable_rows,
        "failed_rows": failed,
        "skipped_coverage_rows": skipped_coverage,
        "num_labels": len(label_to_id),
        "split_counts": split_counts,
        "sample_every": int(args.sample_every),
        "max_frames": int(args.max_frames),
        "outputs": {
            "keypoint_dir": args.keypoint_dir,
            "labels_csv": args.labels_csv,
            "label_map": args.label_map,
        },
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Video->text samples: {usable_rows}")
    print(f"Labels: {len(label_to_id)}")
    print(f"Labels CSV: {args.labels_csv}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
