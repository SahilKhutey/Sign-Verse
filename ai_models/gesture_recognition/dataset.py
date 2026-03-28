"""
Gesture Recognition Dataset

Loads keypoint sequences from .npy files and sign labels from CSV.
Supports WLASL, AUTSL, How2Sign, and PHOENIX-2014T formats.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import csv
import os

from common.keypoint_schema import FEATURE_DIM_225, TWO_HANDS_DIM, HANDS_SLICE_225


class GestureDataset(Dataset):
    """
    Dataset for gesture recognition.

    Directory structure:
        data_dir/
            keypoints/     ← .npy files, shape (frames, feature_dim)
            labels.csv     ← filename, label_id, label_name
    """

    def __init__(self, data_dir, seq_len=30, feature_dim=126, augment=False):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.augment = augment
        self.samples = []

        label_file = os.path.join(data_dir, "labels.csv")
        keypoint_dir = os.path.join(data_dir, "keypoints")

        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for row in csv.DictReader(f):
                    kp_path = os.path.join(keypoint_dir, row["filename"])
                    if os.path.exists(kp_path):
                        self.samples.append((kp_path, int(row["label_id"])))

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            # Return synthetic sample for testing
            seq = np.random.randn(self.seq_len, self.feature_dim).astype(np.float32)
            return torch.tensor(seq), torch.tensor(0)

        kp_path, label = self.samples[idx]
        seq = np.load(kp_path).astype(np.float32)

        # Canonical 225-dim layout is [body(99), left_hand(63), right_hand(63)].
        # If this dataset is configured for 126-dim (hands-only) training, slice out
        # the hands portion rather than taking the first 126 dims (which would
        # include body + partial hand).
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)
        if seq.shape[-1] == FEATURE_DIM_225 and self.feature_dim == TWO_HANDS_DIM:
            seq = seq[:, HANDS_SLICE_225]

        if seq.shape[-1] > self.feature_dim:
            seq = seq[:, :self.feature_dim]
        elif seq.shape[-1] < self.feature_dim:
            seq = np.pad(seq, ((0, 0), (0, self.feature_dim - seq.shape[-1])))

        # Pad / truncate sequence
        if len(seq) >= self.seq_len:
            start = np.random.randint(0, len(seq) - self.seq_len + 1) if self.augment else 0
            seq = seq[start:start + self.seq_len]
        else:
            seq = np.pad(seq, ((0, self.seq_len - len(seq)), (0, 0)))

        if self.augment:
            seq = seq + np.random.randn(*seq.shape).astype(np.float32) * 0.01

        return torch.tensor(seq), torch.tensor(label)
