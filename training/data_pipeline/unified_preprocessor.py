"""
Unified Preprocessor — Converts all raw datasets into a
single standard format for training.

Processing pipeline per dataset:
    Raw videos → Frame extraction → Pose estimation
    → Normalization → .npy keypoint files
    → Unified labels.csv

Output directory structure:
    training-data/
        keypoints/       ← (frames × 225) per video
        labels.csv       ← video_id, label_id, label_name, dataset, split
        label_map.json   ← unified vocabulary across all datasets
        stats.json       ← dataset statistics
"""

import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from training.data_pipeline.dataset_registry import DATASETS
from training.data_pipeline.pose_extractor import PoseExtractor
from dataset_builder.frame_extractor import extract_frames
from common.keypoint_schema import BODY_DIM, ONE_HAND_DIM, FEATURE_DIM_225


OUTPUT_DIR = "training-data"
KEYPOINT_DIR = os.path.join(OUTPUT_DIR, "keypoints")
RAW_VIDEOS_DIR = os.path.join("datasets", "raw_videos")


class UnifiedPreprocessor:

    def __init__(self, config=None):
        self.config = config or {}
        self.pose_extractor = None  # Lazy init (heavy)
        self.label_map = {}
        self.rows = []
        self.next_label = 0

    def _get_pose_extractor(self):
        if self.pose_extractor is None:
            self.pose_extractor = PoseExtractor()
        return self.pose_extractor

    def _get_or_create_label(self, name):
        if name not in self.label_map:
            self.label_map[name] = self.next_label
            self.next_label += 1
        return self.label_map[name]

    def process_wlasl(self, video_dir, annotation_path):
        """Process WLASL dataset (JSON annotations)."""
        print("\n[WLASL] Processing...")

        if not os.path.exists(annotation_path):
            print(f"  Annotations not found: {annotation_path}")
            return 0

        with open(annotation_path) as f:
            data = json.load(f)

        count = 0
        for entry in data:
            gloss = entry["gloss"].upper()
            label_id = self._get_or_create_label(gloss)

            for instance in entry.get("instances", []):
                video_id = instance.get("video_id", "")
                split = instance.get("split", "train")
                video_path = os.path.join(video_dir, f"{video_id}.mp4")

                if not os.path.exists(video_path):
                    continue

                kp = self._process_video(video_path, video_id)
                if kp is not None:
                    self.rows.append({
                        "filename": f"{video_id}.npy",
                        "label_id": label_id,
                        "label_name": gloss,
                        "dataset": "WLASL",
                        "split": split
                    })
                    count += 1

        print(f"  Processed {count} WLASL videos")
        return count

    def process_labeled_directory(self, dataset_name, root_dir):
        """
        Process datasets where subdirectory name = sign class.

        Expected structure:
            root_dir/
                sign_name/
                    video1.mp4
                    video2.mp4
        """
        print(f"\n[{dataset_name}] Processing...")

        if not os.path.exists(root_dir):
            print(f"  Directory not found: {root_dir}")
            return 0

        count = 0
        for label_name in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            label_id = self._get_or_create_label(label_name.upper())

            for fname in os.listdir(label_dir):
                if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
                    continue

                video_path = os.path.join(label_dir, fname)
                video_id = f"{dataset_name}_{label_name}_{os.path.splitext(fname)[0]}"

                kp = self._process_video(video_path, video_id)
                if kp is not None:
                    self.rows.append({
                        "filename": f"{video_id}.npy",
                        "label_id": label_id,
                        "label_name": label_name.upper(),
                        "dataset": dataset_name,
                        "split": "train"
                    })
                    count += 1

        print(f"  Processed {count} {dataset_name} videos")
        return count

    def process_csv_dataset(self, dataset_name, video_dir, csv_path,
                            video_col="video_id", label_col="label",
                            split_col="split"):
        """Process datasets with CSV annotations."""
        print(f"\n[{dataset_name}] Processing (CSV)...")

        if not os.path.exists(csv_path):
            print(f"  CSV not found: {csv_path}")
            return 0

        count = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video_id = row.get(video_col, "")
                label_name = row.get(label_col, "UNKNOWN").upper()
                split = row.get(split_col, "train")

                video_path = os.path.join(video_dir, f"{video_id}.mp4")
                if not os.path.exists(video_path):
                    continue

                label_id = self._get_or_create_label(label_name)
                kp = self._process_video(video_path, f"{dataset_name}_{video_id}")

                if kp is not None:
                    self.rows.append({
                        "filename": f"{dataset_name}_{video_id}.npy",
                        "label_id": label_id,
                        "label_name": label_name,
                        "dataset": dataset_name,
                        "split": split
                    })
                    count += 1

        print(f"  Processed {count} {dataset_name} videos")
        return count

    def _process_video(self, video_path, video_id):
        """
        Extract pose keypoints from a video and save as .npy.
        Returns the keypoint array or None on failure.
        """
        try:
            extractor = self._get_pose_extractor()
            poses = extractor.extract_video(video_path)

            if len(poses) == 0:
                return None

            vectors = [
                np.concatenate([
                    np.array(p["pose"].get("body") or np.zeros(BODY_DIM), dtype=np.float32).reshape(-1)[:BODY_DIM],
                    np.array(p["pose"].get("left_hand") or np.zeros(ONE_HAND_DIM), dtype=np.float32).reshape(-1)[:ONE_HAND_DIM],
                    np.array(p["pose"].get("right_hand") or np.zeros(ONE_HAND_DIM), dtype=np.float32).reshape(-1)[:ONE_HAND_DIM],
                ], axis=0)
                for p in poses
            ]

            seq = np.stack(vectors, axis=0).astype(np.float32)

            # Hard guarantee on feature dim (T, 225)
            if seq.shape[1] != FEATURE_DIM_225:
                seq = seq[:, :FEATURE_DIM_225]
                if seq.shape[1] < FEATURE_DIM_225:
                    seq = np.pad(seq, ((0, 0), (0, FEATURE_DIM_225 - seq.shape[1])))

            # Normalize
            seq = self._normalize(seq)

            # Save
            os.makedirs(KEYPOINT_DIR, exist_ok=True)
            np.save(os.path.join(KEYPOINT_DIR, f"{video_id}.npy"), seq)
            return seq

        except Exception as e:
            print(f"    Error processing {video_path}: {e}")
            return None

    def _normalize(self, seq):
        """Zero-mean, unit-std normalization per feature."""
        mean = seq.mean(axis=0, keepdims=True)
        std = seq.std(axis=0, keepdims=True) + 1e-8
        return (seq - mean) / std

    def save_manifest(self):
        """Save unified labels.csv, label_map.json, and stats.json."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Labels CSV
        with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w", newline="") as f:
            fieldnames = ["filename", "label_id", "label_name", "dataset", "split"]
            if any(isinstance(r, dict) and ("text" in r) for r in self.rows):
                fieldnames.append("text")
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

        # Label map
        with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
            json.dump(self.label_map, f, indent=2)

        # Stats
        from collections import Counter
        dataset_counts = Counter(r["dataset"] for r in self.rows)
        split_counts = Counter(r["split"] for r in self.rows)

        stats = {
            "total_samples": len(self.rows),
            "total_classes": len(self.label_map),
            "per_dataset": dict(dataset_counts),
            "per_split": dict(split_counts),
        }

        with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*50}")
        print(f"  Manifest saved to {OUTPUT_DIR}/")
        print(f"  Total samples:  {len(self.rows):,}")
        print(f"  Total classes:  {len(self.label_map):,}")
        print(f"  Per dataset:    {dict(dataset_counts)}")
        print(f"{'='*50}\n")

    def run_all(self, datasets=None):
        """Run preprocessing for all available datasets."""
        print(f"\n{'='*60}")
        print(f"  SignVerse Unified Preprocessor")
        print(f"  Scanning for dataset files in: {os.path.abspath(RAW_VIDEOS_DIR)}")
        if datasets:
            print(f"  Filtering for: {datasets}")
        print(f"{'='*60}")

        total = 0

        # WLASL
        if not datasets or "WLASL" in datasets:
            wlasl_dir = os.path.join(RAW_VIDEOS_DIR, "WLASL")
            wlasl_ann = os.path.join("datasets", "metadata", "WLASL", "WLASL_v0.3.json")
            if os.path.exists(wlasl_dir):
                total += self.process_wlasl(wlasl_dir, wlasl_ann)

        # Others
        all_others = ["AUTSL", "LSA64", "How2Sign", "RWTH_PHOENIX", "MS_ASL"]
        for ds in all_others:
            if datasets and ds not in datasets:
                continue
            ds_dir = os.path.join(RAW_VIDEOS_DIR, ds)
            if os.path.exists(ds_dir):
                total += self.process_labeled_directory(ds, ds_dir)

        self.save_manifest()
        print(f"Total processed: {total:,} videos")
        return total


if __name__ == "__main__":
    preprocessor = UnifiedPreprocessor()
    preprocessor.run_all()
