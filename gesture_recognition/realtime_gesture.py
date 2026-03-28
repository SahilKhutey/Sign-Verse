"""
RealtimeGestureRecognizer

Small helper for running the trained GestureModel on keypoints.

Note:
  The primary production inference path is `api_server/`, but this stays useful
  for quick local experiments and sanity checks.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ai_models.gesture_recognition.model import GestureModel


def _extract_state(payload):
    if isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        return payload["model_state"]
    return payload


class RealtimeGestureRecognizer:
    """
    Wraps a trained PyTorch gesture model.

    Accepts a single frame vector (D,) or a sequence (T,D) and returns a class ID
    or label string (if labels are provided).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        seq_len: int = 30,
        input_size: int = 126,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = int(seq_len)
        self.input_size = int(input_size)
        self.labels = list(labels) if labels is not None else None

        state = None
        num_classes = len(self.labels) if self.labels else 500

        if model_path and os.path.exists(model_path):
            payload = torch.load(model_path, map_location=self.device)
            state = _extract_state(payload)
            if isinstance(state, dict):
                try:
                    w = state.get("fc.3.weight")
                    if hasattr(w, "shape"):
                        num_classes = int(w.shape[0])
                except Exception:
                    pass

        self.model = GestureModel(input_size=self.input_size, num_classes=num_classes).to(self.device)

        if isinstance(state, dict):
            # Skip any shape-mismatched tensors (e.g., class count differences).
            cur = self.model.state_dict()
            filtered = {}
            for k, v in state.items():
                if k not in cur:
                    continue
                try:
                    if hasattr(v, "shape") and hasattr(cur[k], "shape") and v.shape != cur[k].shape:
                        continue
                except Exception:
                    pass
                filtered[k] = v
            self.model.load_state_dict(filtered, strict=False)

        self.model.eval()

    def predict(self, keypoints: Union[Sequence[float], Sequence[Sequence[float]]]):
        """
        Args:
            keypoints:
              - single-frame vector (D,)
              - sequence (T, D)

        Returns:
            label string if `labels` provided, else class id, else None.
        """
        if keypoints is None:
            return None

        arr = np.array(keypoints, dtype=np.float32)
        if arr.ndim == 1:
            vec = arr.reshape(-1)
            if vec.size < self.input_size:
                vec = np.pad(vec, (0, self.input_size - vec.size))
            vec = vec[: self.input_size]
            seq = np.repeat(vec.reshape(1, 1, -1), repeats=self.seq_len, axis=1)  # (1, seq_len, D)
        elif arr.ndim == 2:
            # Pad/truncate time and feature dims.
            T, D = arr.shape
            if D < self.input_size:
                arr = np.pad(arr, ((0, 0), (0, self.input_size - D)))
            arr = arr[:, : self.input_size]
            if T < self.seq_len:
                arr = np.pad(arr, ((0, self.seq_len - T), (0, 0)))
            arr = arr[: self.seq_len]
            seq = arr.reshape(1, self.seq_len, self.input_size)
        else:
            return None

        x = torch.tensor(seq, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            pred = int(logits.argmax(dim=1).item())

        if self.labels:
            if 0 <= pred < len(self.labels):
                return self.labels[pred]
            return None
        return pred

