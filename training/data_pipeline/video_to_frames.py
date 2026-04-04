"""
Convert isolated sign videos into frame folders.

Input layout:
  <input_dir>/<label>/*.mp4

Output layout:
  <output_dir>/<label>/<video_stem>/frame_000001.jpg
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_video(video_path: str, out_dir: str, max_frames: int = 120, step: int = 2) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % max(1, step) == 0:
                out_path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
                if saved >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/isolated_videos")
    parser.add_argument("--output-dir", default="datasets/video_frames")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--step", type=int, default=2, help="Save every Nth frame")
    parser.add_argument("--metadata", default="training-data/video_frames_manifest.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    rows = []
    total_videos = 0
    total_frames = 0

    for label_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        for vid in sorted(label_dir.iterdir()):
            if vid.suffix.lower() not in VIDEO_EXTS:
                continue
            total_videos += 1
            sample_out = output_dir / label / vid.stem
            n = extract_video(
                str(vid),
                str(sample_out),
                max_frames=args.max_frames,
                step=args.step,
            )
            total_frames += n
            rows.append(
                {
                    "video_path": str(vid),
                    "label": label,
                    "frames_dir": str(sample_out),
                    "num_frames": n,
                }
            )

    os.makedirs(os.path.dirname(args.metadata), exist_ok=True)
    with open(args.metadata, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label", "frames_dir", "num_frames"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed videos: {total_videos}")
    print(f"Extracted frames: {total_frames}")
    print(f"Manifest: {args.metadata}")


if __name__ == "__main__":
    main()
