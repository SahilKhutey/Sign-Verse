"""
Collect alphabet sign images from webcam for custom ASL/ISL-style datasets.

Output layout:
  datasets/asl_alphabet/
    train/<LABEL>/*.jpg
    val/<LABEL>/*.jpg
    capture_log.csv

Hotkeys:
  SPACE  capture one sample
  A      toggle auto-capture mode
  N / ]  next label
  P / [  previous label
  T      force split=train
  V      force split=val
  R      split=random (train/val by --val-ratio)
  U      undo last saved image
  Q      quit
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

# Ensure project root is importable when script is run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DEFAULT_LABELS = ",".join([chr(c) for c in range(ord("A"), ord("Z") + 1)])


@dataclass
class SaveRecord:
    path: str
    label: str
    split: str


def _parse_labels(raw: str) -> List[str]:
    out = []
    for token in (raw or "").replace(";", ",").split(","):
        t = token.strip().upper()
        if t:
            out.append(t)
    if not out:
        raise ValueError("No labels provided.")
    return out


def _count_existing(dataset_dir: str, labels: List[str]) -> Dict[str, Dict[str, int]]:
    counts = {"train": {}, "val": {}}
    for split in ["train", "val"]:
        for label in labels:
            p = os.path.join(dataset_dir, split, label)
            n = 0
            if os.path.isdir(p):
                for name in os.listdir(p):
                    low = name.lower()
                    if low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png"):
                        n += 1
            counts[split][label] = n
    return counts


def _ensure_dirs(dataset_dir: str, labels: List[str]):
    for split in ["train", "val"]:
        for label in labels:
            os.makedirs(os.path.join(dataset_dir, split, label), exist_ok=True)


def _choose_split(split_mode: str, val_ratio: float) -> str:
    if split_mode in {"train", "val"}:
        return split_mode
    return "val" if random.random() < val_ratio else "train"


def _append_log(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_ms",
                "action",
                "label",
                "split",
                "path",
                "camera",
                "image_size",
                "roi_scale",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _draw_overlay(
    frame: np.ndarray,
    label: str,
    split_mode: str,
    auto_mode: bool,
    counts: Dict[str, Dict[str, int]],
):
    total = counts["train"].get(label, 0) + counts["val"].get(label, 0)
    h, _ = frame.shape[:2]
    cv2.putText(
        frame,
        f"Label: {label}  Split: {split_mode.upper()}  Auto: {'ON' if auto_mode else 'OFF'}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 220, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Count: train={counts['train'].get(label, 0)} val={counts['val'].get(label, 0)} total={total}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 255, 80),
        2,
    )
    cv2.putText(
        frame,
        "SPACE:capture  A:auto  N/P:label  T/V/R:split  U:undo  Q:quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=os.path.join("datasets", "asl_alphabet"))
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--start-label", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--roi-scale", type=float, default=0.55)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["random", "train", "val"], default="random")
    parser.add_argument("--auto-interval", type=float, default=0.7)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--prefix", default="img")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--log-csv", default=os.path.join("datasets", "asl_alphabet", "capture_log.csv"))
    args = parser.parse_args()

    labels = _parse_labels(args.labels)
    if args.start_label:
        start_label = args.start_label.strip().upper()
        label_idx = labels.index(start_label) if start_label in labels else 0
    else:
        label_idx = 0

    if args.val_ratio < 0 or args.val_ratio > 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.roi_scale <= 0 or args.roi_scale > 1:
        raise ValueError("--roi-scale must be in (0, 1]")

    _ensure_dirs(args.dataset_dir, labels)
    counts = _count_existing(args.dataset_dir, labels)
    history: List[SaveRecord] = []

    split_mode = args.split_mode
    auto_mode = False
    last_auto_ts = 0.0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    def _save_sample(frame: np.ndarray):
        nonlocal counts
        label = labels[label_idx]
        total_for_label = counts["train"].get(label, 0) + counts["val"].get(label, 0)
        if args.max_per_label > 0 and total_for_label >= args.max_per_label:
            return

        split = _choose_split(split_mode=split_mode, val_ratio=args.val_ratio)
        out_dir = os.path.join(args.dataset_dir, split, label)
        os.makedirs(out_dir, exist_ok=True)

        idx = counts[split].get(label, 0) + 1
        out_path = os.path.join(out_dir, f"{args.prefix}_{idx:06d}.jpg")

        if args.grayscale:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.image_size > 0:
            frame = cv2.resize(frame, (args.image_size, args.image_size))

        ok = cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        if not ok:
            return

        counts[split][label] = idx
        history.append(SaveRecord(path=out_path, label=label, split=split))
        _append_log(
            args.log_csv,
            {
                "timestamp_ms": int(time.time() * 1000),
                "action": "save",
                "label": label,
                "split": split,
                "path": out_path.replace("\\", "/"),
                "camera": args.camera,
                "image_size": args.image_size,
                "roi_scale": args.roi_scale,
            },
        )

    def _undo_last():
        nonlocal counts
        if not history:
            return
        rec = history.pop()
        if os.path.exists(rec.path):
            try:
                os.remove(rec.path)
            except Exception:
                return
        counts[rec.split][rec.label] = max(0, counts[rec.split].get(rec.label, 0) - 1)
        _append_log(
            args.log_csv,
            {
                "timestamp_ms": int(time.time() * 1000),
                "action": "undo",
                "label": rec.label,
                "split": rec.split,
                "path": rec.path.replace("\\", "/"),
                "camera": args.camera,
                "image_size": args.image_size,
                "roi_scale": args.roi_scale,
            },
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            size = int(min(h, w) * args.roi_scale)
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            x2 = x1 + size
            y2 = y1 + size

            roi = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 180, 255), 2)

            _draw_overlay(
                frame=frame,
                label=labels[label_idx],
                split_mode=split_mode,
                auto_mode=auto_mode,
                counts=counts,
            )
            cv2.imshow("Collect Alphabet Dataset", frame)

            now = time.time()
            if auto_mode and (now - last_auto_ts) >= max(0.05, args.auto_interval):
                _save_sample(roi)
                last_auto_ts = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                _save_sample(roi)
            elif key == ord("a"):
                auto_mode = not auto_mode
                last_auto_ts = now
            elif key in (ord("n"), ord("]")):
                label_idx = (label_idx + 1) % len(labels)
            elif key in (ord("p"), ord("[")):
                label_idx = (label_idx - 1) % len(labels)
            elif key == ord("t"):
                split_mode = "train"
            elif key == ord("v"):
                split_mode = "val"
            elif key == ord("r"):
                split_mode = "random"
            elif key == ord("u"):
                _undo_last()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Collection session ended.")
    for label in labels:
        train_n = counts["train"].get(label, 0)
        val_n = counts["val"].get(label, 0)
        total_n = train_n + val_n
        if total_n > 0:
            print(f"{label}: train={train_n}, val={val_n}, total={total_n}")


if __name__ == "__main__":
    main()
