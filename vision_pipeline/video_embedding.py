"""
Video embedding utilities using MediaPipe-derived frame features.

Embedding format:
  - mean(frame_features) concatenated with std(frame_features)
  - default dim: 225 * 2 = 450
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np

from vision_pipeline.feature_extractor import FeatureExtractor


class VideoEmbeddingExtractor:
    def __init__(
        self,
        sample_every: int = 2,
        max_frames: int = 120,
        allow_mock: bool = False,
    ):
        self.sample_every = max(1, int(sample_every))
        self.max_frames = max(1, int(max_frames))
        self.extractor = FeatureExtractor()
        if getattr(self.extractor, "use_mock", False) and not allow_mock:
            raise RuntimeError(
                "MediaPipe extractor unavailable (mock mode). Install mediapipe or pass allow_mock=True."
            )

    def _sample_indices(self, n: int) -> np.ndarray:
        if n <= self.max_frames:
            return np.arange(n, dtype=np.int32)
        return np.linspace(0, n - 1, self.max_frames).astype(np.int32)

    def extract_frame_features(self, video_path: str) -> np.ndarray:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        feats = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.sample_every == 0:
                    f, _, _ = self.extractor.extract(frame)
                    feats.append(np.asarray(f, dtype=np.float32))
                idx += 1
        finally:
            cap.release()

        if not feats:
            return np.zeros((0, 225), dtype=np.float32)

        arr = np.stack(feats, axis=0)
        keep = self._sample_indices(arr.shape[0])
        return arr[keep]

    def embed_video(self, video_path: str) -> Dict[str, object]:
        frame_feats = self.extract_frame_features(video_path)
        if frame_feats.shape[0] == 0:
            embedding = np.zeros((450,), dtype=np.float32)
            return {
                "embedding": embedding,
                "num_frames": 0,
                "feature_dim": 225,
                "embedding_dim": int(embedding.shape[0]),
            }

        mean = frame_feats.mean(axis=0)
        std = frame_feats.std(axis=0)
        embedding = np.concatenate([mean, std], axis=0).astype(np.float32)
        return {
            "embedding": embedding,
            "num_frames": int(frame_feats.shape[0]),
            "feature_dim": int(frame_feats.shape[1]),
            "embedding_dim": int(embedding.shape[0]),
            "frame_features": frame_feats,
        }
