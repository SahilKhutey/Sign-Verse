"""
Keypoint Processing Utilities

Normalization and smoothing helpers for keypoint vectors/sequences.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-std normalization."""
    v = vec.astype(np.float32, copy=False)
    return (v - v.mean()) / (v.std() + eps)


def normalize_sequence(seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each feature dimension across time for a (T, D) sequence."""
    x = seq.astype(np.float32, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + eps
    return (x - mean) / std
