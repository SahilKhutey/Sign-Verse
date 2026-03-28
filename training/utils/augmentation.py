"""
Data Augmentation Utilities (Keypoints)

These helpers are intentionally simple and safe defaults for sign keypoint
sequences. They are used by datasets/trainers that want lightweight
regularization without adding new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _is_torch(x: ArrayLike) -> bool:
    return torch is not None and hasattr(torch, "Tensor") and isinstance(x, torch.Tensor)


def jitter(x: ArrayLike, sigma: float = 0.01) -> ArrayLike:
    """Add small Gaussian noise to coordinates."""
    if sigma <= 0:
        return x
    if _is_torch(x):
        return x + torch.randn_like(x) * float(sigma)
    return x + np.random.randn(*x.shape).astype(np.float32) * float(sigma)


def random_scale(x: ArrayLike, min_scale: float = 0.95, max_scale: float = 1.05) -> ArrayLike:
    """Multiply all features by a random scalar."""
    if max_scale <= 0:
        return x
    s = np.random.uniform(min_scale, max_scale)
    return x * float(s)


def dropout_frames(seq: ArrayLike, p: float = 0.05) -> ArrayLike:
    """
    Randomly zero out some frames in a (T, D) sequence.
    """
    if p <= 0:
        return seq
    if _is_torch(seq):
        T = seq.size(0)
        mask = (torch.rand(T, device=seq.device) > float(p)).to(seq.dtype).unsqueeze(1)
        return seq * mask
    T = seq.shape[0]
    mask = (np.random.rand(T) > float(p)).astype(np.float32)[:, None]
    return seq * mask


@dataclass
class KeypointAugmenter:
    sigma: float = 0.01
    min_scale: float = 0.95
    max_scale: float = 1.05
    frame_dropout_p: float = 0.05

    def __call__(self, seq: ArrayLike) -> ArrayLike:
        out = seq
        out = jitter(out, sigma=self.sigma)
        out = random_scale(out, min_scale=self.min_scale, max_scale=self.max_scale)
        out = dropout_frames(out, p=self.frame_dropout_p)
        return out
