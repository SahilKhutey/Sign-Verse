"""
Video frame-folder dataset for clean-room Conv3D isolated sign research baseline.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _deterministic_split(key: str, train_ratio: float, val_ratio: float) -> str:
    train_ratio = max(0.0, min(1.0, float(train_ratio)))
    val_ratio = max(0.0, min(1.0, float(val_ratio)))
    total = max(1e-6, train_ratio + val_ratio)
    train_cut = train_ratio / total
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / float(0xFFFFFFFF)
    return "train" if bucket < train_cut else "val"


class Conv3DVideoDataset(Dataset):
    """
    Reads rows from video frame manifest and returns tensors:
      x: (3, T, H, W), y: scalar class id
    """

    def __init__(
        self,
        manifest_csv: str = os.path.join("training-data", "video_frames_manifest.csv"),
        seq_len: int = 24,
        image_size: int = 112,
        split: str = "train",
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        augment: bool = False,
    ):
        self.manifest_csv = manifest_csv
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.split = split
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.augment = bool(augment)

        self.samples: List[Tuple[str, int]] = []
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.manifest_csv):
            return

        rows = []
        with open(self.manifest_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                frames_dir = (row.get("frames_dir") or "").strip()
                label = (row.get("label") or "").strip()
                if not frames_dir or not label:
                    continue
                if not os.path.isdir(frames_dir):
                    continue
                row_split = (row.get("split") or "").strip().lower()
                if row_split not in {"train", "val"}:
                    key = f"{frames_dir}|{label}"
                    row_split = _deterministic_split(
                        key=key,
                        train_ratio=self.train_ratio,
                        val_ratio=self.val_ratio,
                    )
                rows.append((frames_dir, label, row_split))

        labels_sorted = sorted({label for _, label, _ in rows})
        self.label_to_id = {label: i for i, label in enumerate(labels_sorted)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        for frames_dir, label, row_split in rows:
            if row_split != self.split:
                continue
            self.samples.append((frames_dir, self.label_to_id[label]))

    def __len__(self):
        return max(1, len(self.samples))

    def _load_frames(self, frames_dir: str) -> np.ndarray:
        paths = sorted(
            str(p) for p in Path(frames_dir).glob("*.jpg")
        )
        if not paths:
            return np.zeros((self.seq_len, self.image_size, self.image_size, 3), dtype=np.float32)

        if len(paths) > self.seq_len:
            idx = np.linspace(0, len(paths) - 1, self.seq_len).astype(int).tolist()
            paths = [paths[i] for i in idx]

        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            frames.append(img.astype(np.float32) / 255.0)

        if not frames:
            arr = np.zeros((1, self.image_size, self.image_size, 3), dtype=np.float32)
        else:
            arr = np.stack(frames, axis=0)

        if arr.shape[0] < self.seq_len:
            pad = np.zeros(
                (self.seq_len - arr.shape[0], self.image_size, self.image_size, 3),
                dtype=np.float32,
            )
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > self.seq_len:
            arr = arr[: self.seq_len]

        if self.augment:
            if np.random.rand() < 0.5:
                arr = arr[:, :, ::-1, :]
            arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * 0.01, 0.0, 1.0)

        return arr

    def __getitem__(self, idx: int):
        if len(self.samples) == 0:
            x = np.random.randn(3, self.seq_len, self.image_size, self.image_size).astype(np.float32)
            return torch.tensor(x), torch.tensor(0, dtype=torch.long)

        frames_dir, label_id = self.samples[idx]
        arr = self._load_frames(frames_dir)  # (T,H,W,3)
        arr = np.transpose(arr, (3, 0, 1, 2))  # (C,T,H,W)
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(int(label_id), dtype=torch.long)
