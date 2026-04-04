"""
MultimodalDataset — Real DataLoader for the Multimodal Sign Language LLM.

Loads paired (gesture_keypoints, text_tokens) samples for training.

Data layout expected:
    training-data/
        keypoints/         *.npy  — shape (T, 225) float32
        labels.csv         — columns: filename, label_id, label_name, dataset, split
    training-data/vocab.json  — optional {word: id} text vocabulary

If only gesture data is available (no paired text), falls back to
self-supervised mode: text_target = gesture_token_ids (masked gesture modeling).

Usage:
    dataset = MultimodalDataset(
        keypoint_dir="training-data/keypoints",
        labels_csv="training-data/labels.csv",
        seq_len=60,
        text_seq_len=25,
        split="train"
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for gesture, text_input, text_target in loader:
        ...
"""

import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Tuple

from common.keypoint_schema import FEATURE_DIM_225, TWO_HANDS_DIM, HANDS_SLICE_225


class MultimodalDataset(Dataset):
    """
    Dataset for Multimodal Sign LLM training.

    Each sample returns:
        gesture_seq:    (seq_len, feature_dim) float32 tensor
        text_input:     (text_seq_len,) int64 token ids (BOS + context)
        text_target:    (text_seq_len,) int64 token ids (next-token targets)
    """

    # Special token IDs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(
        self,
        keypoint_dir: str = "training-data/keypoints",
        labels_csv: str = "training-data/labels.csv",
        vocab_path: Optional[str] = "training-data/vocab.json",
        seq_len: int = 60,
        text_seq_len: int = 25,
        feature_dim: int = 225,
        split: str = "train",
        augment: bool = False,
    ):
        self.keypoint_dir = keypoint_dir
        self.seq_len = seq_len
        self.text_seq_len = text_seq_len
        self.feature_dim = feature_dim
        self.augment = augment
        self.split = split

        # Load vocabulary
        self.vocab = self._load_vocab(vocab_path)
        self.vocab_size = max(self.vocab.values(), default=100) + 1

        # Load samples
        self.samples = self._load_samples(labels_csv, split)

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}' in {labels_csv}. "
                f"Run the preprocessing pipeline first."
            )

        print(f"  MultimodalDataset [{split}]: {len(self.samples)} samples | "
              f"seq_len={seq_len} | text_seq_len={text_seq_len}")

    # ──────────────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.samples[idx]
        kp_path = row["kp_path"]
        label_id = row["label_id"]
        label_name = row["label_name"]

        # ── Gesture sequence ──────────────────────────────────────
        gesture_seq = self._load_keypoints(kp_path)

        # ── Text tokens ───────────────────────────────────────────
        text_tokens = self._label_to_tokens(label_name)

        # Build teacher-forced input/target pair
        # input:  [BOS, t1, t2, ..., t_{n-1}]  (length = text_seq_len)
        # target: [t1,  t2, ..., t_n,   EOS]   (length = text_seq_len)
        text_input, text_target = self._build_lm_pair(text_tokens)

        return gesture_seq, text_input, text_target

    # ──────────────────────────────────────────────────────────────
    # Data loading helpers
    # ──────────────────────────────────────────────────────────────

    def _load_keypoints(self, path: str) -> torch.Tensor:
        """Load .npy keypoints, normalize, pad/truncate to seq_len."""
        if os.path.exists(path):
            arr = np.load(path).astype(np.float32)
        else:
            arr = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)

        # Ensure 2D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Pad / truncate to feature_dim on axis=1
        T, D = arr.shape

        # If a caller configures a hands-only feature_dim (126) but the stored
        # file uses the canonical 225-dim layout, slice out the hands portion.
        if D == FEATURE_DIM_225 and self.feature_dim == TWO_HANDS_DIM:
            arr = arr[:, HANDS_SLICE_225]
            T, D = arr.shape

        if D < self.feature_dim:
            arr = np.pad(arr, ((0, 0), (0, self.feature_dim - D)))
        else:
            arr = arr[:, :self.feature_dim]

        # Pad / truncate to seq_len on axis=0
        if T < self.seq_len:
            arr = np.pad(arr, ((0, self.seq_len - T), (0, 0)))
        else:
            arr = arr[:self.seq_len]

        # Normalize per-frame
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-6
        arr = (arr - mean) / std

        # Optional augmentation
        if self.augment:
            arr = self._augment(arr)

        return torch.tensor(arr, dtype=torch.float32)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Apply random augmentations to keypoint sequence."""
        # Gaussian noise
        arr = arr + np.random.randn(*arr.shape).astype(np.float32) * 0.02
        # Temporal jitter (random crop within seq_len)
        if arr.shape[0] > 10:
            start = np.random.randint(0, min(5, arr.shape[0] // 4))
            end = arr.shape[0] - np.random.randint(0, min(5, arr.shape[0] // 4))
            crop = arr[start:end]
            # Resize back (simple linear interpolation)
            indices = np.linspace(0, len(crop) - 1, self.seq_len)
            arr = np.array([crop[int(i)] for i in indices])
        # Scale jitter
        arr = arr * np.random.uniform(0.95, 1.05)
        return arr.astype(np.float32)

    def _label_to_tokens(self, label_name: str) -> List[int]:
        """Convert label name to list of token IDs."""
        # Split by underscore to get words, e.g. "THANK_YOU" → ["THANK", "YOU"]
        words = label_name.upper().split("_")
        tokens = []
        for word in words:
            tokens.append(self.vocab.get(word, self.UNK_ID))
        return tokens

    def _build_lm_pair(
        self, tokens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build teacher-forcing input/target pair for LM training."""
        L = self.text_seq_len

        # Prepend BOS, append EOS
        full = [self.BOS_ID] + tokens[:L - 1] + [self.EOS_ID]

        # Input: first L tokens (may be padded)
        text_input = np.full(L, self.PAD_ID, dtype=np.int64)
        src = full[:L]
        text_input[:len(src)] = src

        # Target: tokens shifted left by 1 (next-token prediction)
        text_target = np.full(L, self.PAD_ID, dtype=np.int64)
        tgt = full[1:L + 1]
        text_target[:len(tgt)] = tgt

        return (torch.tensor(text_input, dtype=torch.long),
                torch.tensor(text_target, dtype=torch.long))

    # ──────────────────────────────────────────────────────────────
    # Sample loading
    # ──────────────────────────────────────────────────────────────

    def _load_samples(self, labels_csv: str, split: str) -> list:
        """Parse labels.csv and return list of sample dicts."""
        if not os.path.exists(labels_csv):
            return []

        samples = []
        with open(labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get("split", "train")
                if row_split != split:
                    continue

                filename = row.get("filename", "")
                kp_path = os.path.join(self.keypoint_dir, filename)

                try:
                    label_id = int(row.get("label_id", 0))
                except ValueError:
                    label_id = 0

                samples.append({
                    "kp_path": kp_path,
                    "label_id": label_id,
                    "label_name": row.get("label_name", f"SIGN_{label_id}"),
                    "dataset": row.get("dataset", "UNKNOWN")
                })

        return samples

    # ──────────────────────────────────────────────────────────────
    # Vocabulary
    # ──────────────────────────────────────────────────────────────

    def _load_vocab(self, vocab_path: Optional[str]) -> dict:
        """Load vocabulary from JSON or create minimal default."""
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                return json.load(f)

        # Minimal default vocab (will be overridden by label-based tokens)
        return {
            "<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3
        }

    @classmethod
    def build_vocab_from_labels(cls, labels_csv: str,
                                 save_path: Optional[str] = None) -> dict:
        """
        Build vocabulary from all label names in labels.csv.
        Saves to save_path if provided.
        """
        vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        idx = 4

        if not os.path.exists(labels_csv):
            return vocab

        with open(labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("label_name", "")
                for word in name.upper().split("_"):
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(vocab, f, indent=2)
            print(f"  Vocabulary ({len(vocab)} tokens) → {save_path}")

        return vocab


# ──────────────────────────────────────────────────────────────────
# Synthetic dataset (for testing without real data)
# ──────────────────────────────────────────────────────────────────

class SyntheticMultimodalDataset(Dataset):
    """
    Generates random gesture + text pairs for pipeline testing.
    No files required; drop-in replacement for MultimodalDataset.
    """

    def __init__(self, n_samples: int = 200, seq_len: int = 60,
                 feature_dim: int = 225, text_seq_len: int = 25,
                 vocab_size: int = 10000):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.text_seq_len = text_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        gesture = torch.randn(self.seq_len, self.feature_dim)
        text_input = torch.randint(0, self.vocab_size, (self.text_seq_len,))
        text_target = torch.randint(0, self.vocab_size, (self.text_seq_len,))
        return gesture, text_input, text_target
