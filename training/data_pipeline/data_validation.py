"""
Data Validation — Checks dataset integrity before training.

Validates:
    - Label files exist and are readable
    - Keypoint files exist and have correct shape
    - No corrupt or zero-variance sequences
    - Class distribution balance
    - Train/val/test split ratios
"""

import os
import sys
import csv
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def validate_dataset(data_dir="training-data", feature_dim=225, verbose=True):
    """
    Full dataset validation report.

    Returns:
        dict with validation results and any warnings
    """
    issues = []
    warnings = []
    stats = {}

    labels_path = os.path.join(data_dir, "labels.csv")
    keypoint_dir = os.path.join(data_dir, "keypoints")
    label_map_path = os.path.join(data_dir, "label_map.json")

    print(f"\n{'='*60}")
    print(f"  SignVerse Data Validator")
    print(f"  Data directory: {os.path.abspath(data_dir)}")
    print(f"{'='*60}")

    # If label_map.json is missing (common for older/synthetic pipelines), try to
    # synthesize it from labels.csv so downstream steps can run.
    if (not os.path.exists(label_map_path)) and os.path.exists(labels_path):
        try:
            label_map = {}
            with open(labels_path, "r", newline="") as f:
                for row in csv.DictReader(f):
                    name = (row.get("label_name") or row.get("label") or row.get("gloss") or "").strip()
                    if not name:
                        continue
                    if name in label_map:
                        continue
                    lid = row.get("label_id")
                    try:
                        label_map[name] = int(lid)
                    except Exception:
                        label_map[name] = len(label_map)
            if label_map:
                with open(label_map_path, "w", encoding="utf-8") as f:
                    json.dump(label_map, f, indent=2)
                warnings.append("Generated missing label_map.json from labels.csv")
        except Exception as e:
            warnings.append(f"Could not generate label_map.json: {e}")

    # ── 1. Check required files exist ──────────────────────────
    for path in [labels_path, keypoint_dir, label_map_path]:
        if not os.path.exists(path):
            issues.append(f"Missing: {path}")

    if issues:
        print("\n  [X] Missing files:")
        for issue in issues:
            print(f"    - {issue}")
        return {"valid": False, "issues": issues}

    # ── 2. Load label map ───────────────────────────────────────
    with open(label_map_path) as f:
        label_map = json.load(f)
    stats["num_classes"] = len(label_map)

    # ── 3. Parse labels CSV ─────────────────────────────────────
    rows = []
    class_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    split_counts = defaultdict(int)

    with open(labels_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            class_counts[row["label_name"]] += 1
            dataset_counts[row.get("dataset", "unknown")] += 1
            split_counts[row.get("split", "train")] += 1

    stats["total_samples"] = len(rows)
    stats["dataset_counts"] = dict(dataset_counts)
    stats["split_counts"] = dict(split_counts)

    # ── 4. Validate keypoint files ──────────────────────────────
    missing_files = []
    corrupt_files = []
    shape_errors = []
    zero_variance = []
    valid_count = 0

    print(f"\n  Checking {len(rows):,} keypoint files...")

    for i, row in enumerate(rows):
        kp_path = os.path.join(keypoint_dir, row["filename"])

        if not os.path.exists(kp_path):
            missing_files.append(row["filename"])
            continue

        try:
            arr = np.load(kp_path)

            if arr.ndim != 2:
                shape_errors.append(f"{row['filename']}: ndim={arr.ndim}")
                continue

            if arr.shape[-1] < feature_dim:
                shape_errors.append(
                    f"{row['filename']}: dim={arr.shape[-1]} (expected {feature_dim})"
                )

            if arr.std() < 1e-6:
                zero_variance.append(row["filename"])

            valid_count += 1

        except Exception as e:
            corrupt_files.append(f"{row['filename']}: {e}")

        if verbose and (i + 1) % 500 == 0:
            print(f"    Checked {i+1:,} / {len(rows):,}...")

    stats["valid_files"] = valid_count
    stats["missing_files"] = len(missing_files)
    stats["corrupt_files"] = len(corrupt_files)
    stats["shape_errors"] = len(shape_errors)
    stats["zero_variance_files"] = len(zero_variance)

    if missing_files:
        warnings.append(f"{len(missing_files)} missing keypoint files")
    if corrupt_files:
        warnings.append(f"{len(corrupt_files)} corrupt files")
    if shape_errors:
        warnings.append(f"{len(shape_errors)} shape mismatches")
    if zero_variance:
        warnings.append(f"{len(zero_variance)} zero-variance sequences")

    # ── 5. Class imbalance check ────────────────────────────────
    counts = list(class_counts.values())
    if counts:
        max_c = max(counts)
        min_c = min(counts)
        imbalance_ratio = max_c / max(min_c, 1)
        stats["class_imbalance_ratio"] = round(imbalance_ratio, 2)
        if imbalance_ratio > 10:
            warnings.append(
                f"High class imbalance: {imbalance_ratio:.1f}x "
                f"(max={max_c}, min={min_c})"
            )

    # ── 6. Split ratio check ────────────────────────────────────
    total = max(stats["total_samples"], 1)
    train_ratio = split_counts.get("train", 0) / total
    if train_ratio < 0.6:
        warnings.append(f"Low train ratio: {train_ratio:.1%}")

    # ── 7. Print report ─────────────────────────────────────────
    valid = len(issues) == 0

    print(f"\n  Results:")
    print(f"    Total samples:      {stats['total_samples']:>10,}")
    print(f"    Valid files:        {stats['valid_files']:>10,}")
    print(f"    Classes:            {stats['num_classes']:>10,}")
    print(f"    Missing files:      {stats['missing_files']:>10,}")
    print(f"    Corrupt files:      {stats['corrupt_files']:>10,}")
    print(f"    Shape errors:       {stats['shape_errors']:>10,}")
    print(f"    Zero-variance:      {stats['zero_variance_files']:>10,}")
    print(f"    Imbalance ratio:    {stats.get('class_imbalance_ratio', 'N/A'):>10}")
    print(f"\n  Datasets:")
    for ds, cnt in stats["dataset_counts"].items():
        print(f"    {ds:<20} {cnt:>8,}")
    print(f"\n  Splits:")
    for sp, cnt in stats["split_counts"].items():
        print(f"    {sp:<10} {cnt:>8,} ({cnt/total:.1%})")

    if warnings:
        print(f"\n  [!] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    if issues:
        print(f"\n  [X] Issues ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
        valid = False
    else:
        print("\n  [OK] Dataset validated successfully!")

    print(f"{'='*60}\n")

    return {
        "valid": valid,
        "stats": stats,
        "warnings": warnings,
        "issues": issues,
        "missing": missing_files[:20],
        "corrupt": corrupt_files[:10],
    }


if __name__ == "__main__":
    result = validate_dataset()
    if not result["valid"]:
        print("Fix issues before training.")
        exit(1)
    print("Ready to train!")
