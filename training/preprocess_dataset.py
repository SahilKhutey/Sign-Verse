"""
Dataset Preprocessing Entrypoint

Converts raw videos under `datasets/raw_videos/` into the unified training
format under `training-data/` using `UnifiedPreprocessor`.

Usage:
  python training/preprocess_dataset.py
  python training/preprocess_dataset.py --data-dir training-data
"""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets into training-data/")
    parser.add_argument("--data-dir", default="training-data", help="Output directory")
    args = parser.parse_args()

    import training.data_pipeline.unified_preprocessor as up

    # Override module-level output paths used by UnifiedPreprocessor.
    if args.data_dir != "training-data":
        up.OUTPUT_DIR = args.data_dir
        up.KEYPOINT_DIR = os.path.join(up.OUTPUT_DIR, "keypoints")

    pre = up.UnifiedPreprocessor()
    pre.run_all()
    pre.save_manifest()


if __name__ == "__main__":
    main()
