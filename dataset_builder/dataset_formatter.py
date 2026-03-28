"""
Dataset Builder — Formatter

Aggregates keypoint sequences from multiple videos into
a unified training dataset format.

Output:
    training-data/
        keypoints/     ← per-video .npy files (frames × 225)
        labels.csv     ← video_id, label_id, label_name, split
"""

import os
import csv
import json
import numpy as np


def build_dataset(pose_root, output_dir, label_map=None):
    """
    Walk all subdirectories, stack per-frame keypoints
    into per-video sequences, and write labels.csv.

    Directory structure expected:
        pose_root/
            sign_name/
                0000.npy, 0001.npy, ...
    """
    os.makedirs(os.path.join(output_dir, "keypoints"), exist_ok=True)

    label_map = label_map or {}
    rows = []
    next_label = len(label_map)

    for label_name in sorted(os.listdir(pose_root)):
        label_dir = os.path.join(pose_root, label_name)
        if not os.path.isdir(label_dir):
            continue

        if label_name not in label_map:
            label_map[label_name] = next_label
            next_label += 1

        label_id = label_map[label_name]

        # Stack all frames into sequence
        frames = []
        for nf in sorted(os.listdir(label_dir)):
            if nf.endswith('.npy'):
                arr = np.load(os.path.join(label_dir, nf))
                frames.append(arr)

        if not frames:
            continue

        seq = np.stack(frames, axis=0)  # (frames, 225)
        out_name = f"{label_name}_seq.npy"
        np.save(os.path.join(output_dir, "keypoints", out_name), seq)
        rows.append({"filename": out_name, "label_id": label_id, "label_name": label_name})

    # Write labels CSV
    with open(os.path.join(output_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label_id", "label_name"])
        writer.writeheader()
        writer.writerows(rows)

    # Save label map
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Dataset built: {len(rows)} sequences | {len(label_map)} classes")
    return label_map
