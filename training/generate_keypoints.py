"""
Generate Keypoints (Video -> .npy)

Extracts 225-dim per-frame keypoints (body + 2 hands) from a single video.

Usage:
  python training/generate_keypoints.py --video path/to/video.mp4 --out training-data/keypoints/sample.npy
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from common.keypoint_schema import BODY_DIM, ONE_HAND_DIM, FEATURE_DIM_225


def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    from training.data_pipeline.pose_extractor import PoseExtractor

    extractor = PoseExtractor()
    poses = extractor.extract_video(video_path)

    vectors = []
    for p in poses:
        pose = p.get("pose", {})
        vec = np.concatenate(
            [
                np.array(pose.get("body") or np.zeros(BODY_DIM), dtype=np.float32).reshape(-1)[:BODY_DIM],
                np.array(pose.get("left_hand") or np.zeros(ONE_HAND_DIM), dtype=np.float32).reshape(-1)[:ONE_HAND_DIM],
                np.array(pose.get("right_hand") or np.zeros(ONE_HAND_DIM), dtype=np.float32).reshape(-1)[:ONE_HAND_DIM],
            ],
            axis=0,
        )
        if vec.shape[0] != FEATURE_DIM_225:
            vec = vec[:FEATURE_DIM_225]
            if vec.shape[0] < FEATURE_DIM_225:
                vec = np.pad(vec, (0, FEATURE_DIM_225 - vec.shape[0]))
        vectors.append(vec)

    if not vectors:
        return np.zeros((0, FEATURE_DIM_225), dtype=np.float32)

    return np.stack(vectors, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract 225-dim keypoints from a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Path to output .npy file")
    args = parser.parse_args()

    seq = extract_keypoints_from_video(args.video)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, seq)
    print(f"Saved keypoints: {args.out}  shape={seq.shape}")


if __name__ == "__main__":
    main()
