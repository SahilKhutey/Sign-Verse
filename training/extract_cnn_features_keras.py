"""
Extract frame-level CNN features (InceptionV3) for video samples.

Input:
  --manifest training-data/video_frames_manifest.csv

Output:
  --output-dir training-data/video_features
  --output-manifest training-data/video_features_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


def _sample_paths(frame_paths, max_frames: int):
    if len(frame_paths) <= max_frames:
        return frame_paths
    idx = np.linspace(0, len(frame_paths) - 1, max_frames).astype(int).tolist()
    return [frame_paths[i] for i in idx]


def _load_frame(path: str, image_size: int):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    return img.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training-data/video_frames_manifest.csv")
    parser.add_argument("--output-dir", default="training-data/video_features")
    parser.add_argument("--output-manifest", default="training-data/video_features_manifest.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    try:
        from tensorflow.keras.applications import InceptionV3  # type: ignore
        from tensorflow.keras.applications.inception_v3 import preprocess_input  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow/Keras not installed. Install optional ASL CNN dependencies."
        ) from exc

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    model = InceptionV3(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(args.image_size, args.image_size, 3),
    )

    rows_out = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames_dir = row.get("frames_dir", "")
            label = row.get("label", "")
            if not frames_dir or not os.path.isdir(frames_dir):
                continue
            frame_paths = sorted(
                str(p) for p in Path(frames_dir).glob("*.jpg")
            )
            if not frame_paths:
                continue
            frame_paths = _sample_paths(frame_paths, args.max_frames)

            imgs = []
            for fp in frame_paths:
                img = _load_frame(fp, args.image_size)
                if img is not None:
                    imgs.append(img)
            if not imgs:
                continue

            x = np.array(imgs, dtype=np.float32)
            x = preprocess_input(x)
            feats = model.predict(x, verbose=0)  # (T, 2048)

            sample_id = os.path.basename(frames_dir)
            out_dir = os.path.join(args.output_dir, label)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{sample_id}.npy")
            np.save(out_path, feats.astype(np.float32))

            rows_out.append(
                {
                    "features_path": out_path,
                    "label": label,
                    "num_frames": feats.shape[0],
                    "feature_dim": feats.shape[1] if feats.ndim == 2 else 0,
                }
            )

    os.makedirs(os.path.dirname(args.output_manifest), exist_ok=True)
    with open(args.output_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["features_path", "label", "num_frames", "feature_dim"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Extracted features for {len(rows_out)} samples")
    print(f"Feature manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
