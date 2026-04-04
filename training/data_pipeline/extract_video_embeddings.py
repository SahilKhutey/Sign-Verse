"""
Extract MediaPipe-based embeddings for sign videos.

Input layout:
  <input_dir>/<label>/*.mp4

Outputs:
  - .npy embedding files in output_dir/<label>/
  - CSV manifest with paths and metadata
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vision_pipeline.video_embedding import VideoEmbeddingExtractor


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=os.path.join("datasets", "isolated_videos"))
    parser.add_argument("--output-dir", default=os.path.join("training-data", "video_embeddings"))
    parser.add_argument(
        "--manifest",
        default=os.path.join("training-data", "video_embeddings_manifest.csv"),
    )
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--save-frame-features", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    extractor = VideoEmbeddingExtractor(
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        allow_mock=args.allow_mock,
    )

    rows = []
    processed = 0
    failed = 0

    for label_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        for vid in sorted(label_dir.iterdir()):
            if not vid.is_file() or vid.suffix.lower() not in VIDEO_EXTS:
                continue

            out_label_dir = Path(args.output_dir) / label
            out_label_dir.mkdir(parents=True, exist_ok=True)
            emb_path = out_label_dir / f"{vid.stem}_embed.npy"
            frame_feat_path = out_label_dir / f"{vid.stem}_frames.npy"
            try:
                out = extractor.embed_video(str(vid))
                np.save(emb_path, out["embedding"])
                if args.save_frame_features:
                    np.save(frame_feat_path, out.get("frame_features", np.zeros((0, 225), dtype=np.float32)))
                rows.append(
                    {
                        "video_path": str(vid).replace("\\", "/"),
                        "label": label,
                        "embedding_path": str(emb_path).replace("\\", "/"),
                        "frame_features_path": str(frame_feat_path).replace("\\", "/")
                        if args.save_frame_features
                        else "",
                        "num_frames": int(out.get("num_frames") or 0),
                        "feature_dim": int(out.get("feature_dim") or 0),
                        "embedding_dim": int(out.get("embedding_dim") or 0),
                    }
                )
                processed += 1
            except Exception:
                failed += 1
                continue

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "label",
                "embedding_path",
                "frame_features_path",
                "num_frames",
                "feature_dim",
                "embedding_dim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed videos: {processed}")
    print(f"Failed videos: {failed}")
    print(f"Embedding manifest: {args.manifest}")


if __name__ == "__main__":
    main()
