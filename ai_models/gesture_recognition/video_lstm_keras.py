"""
Video CNN+LSTM Classifier (Keras)

Pipeline:
  video -> sampled RGB frames -> InceptionV3 feature vectors -> LSTM classifier
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

import cv2
import numpy as np


class VideoLSTMClassifier:
    def __init__(
        self,
        model_path: str,
        labels_path: str | None = None,
        image_size: int = 224,
        max_frames: int = 30,
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.image_size = int(image_size)
        self.max_frames = int(max_frames)

        self.model = None
        self.feature_extractor = None
        self.labels: List[str] = []
        self._load()

    def _load(self):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            from tensorflow.keras.applications import InceptionV3  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow/Keras not available. Install optional dependencies for video LSTM."
            ) from exc

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Video LSTM model not found: {self.model_path}")

        self.model = load_model(self.model_path)
        self.feature_extractor = InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(self.image_size, self.image_size, 3),
        )

        if self.labels_path and os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.labels = [str(x) for x in data]
            elif isinstance(data, dict):
                self.labels = [str(v) for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]

    def _read_video_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
        return frames

    def _sample_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        if len(frames) <= self.max_frames:
            return frames
        idx = np.linspace(0, len(frames) - 1, self.max_frames).astype(int).tolist()
        return [frames[i] for i in idx]

    def _preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("No frames provided")
        arr = []
        for frame in frames:
            x = cv2.resize(frame, (self.image_size, self.image_size)).astype(np.float32)
            arr.append(x)
        x = np.array(arr, dtype=np.float32)

        from tensorflow.keras.applications.inception_v3 import preprocess_input  # type: ignore
        x = preprocess_input(x)
        return x

    def _extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not loaded")
        x = self._preprocess_frames(frames)
        feats = self.feature_extractor.predict(x, verbose=0)  # (T, 2048)
        return feats

    def _pad_or_truncate(self, feats: np.ndarray) -> np.ndarray:
        if feats.ndim != 2:
            raise ValueError("Features must be 2D (T, D)")
        t, d = feats.shape
        if t < self.max_frames:
            pad = np.zeros((self.max_frames - t, d), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        elif t > self.max_frames:
            feats = feats[: self.max_frames]
        return feats

    def predict_video_path(self, video_path: str) -> Dict[str, float | int | str | None]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        frames = self._read_video_frames(video_path)
        frames = self._sample_frames(frames)
        if not frames:
            return {"class_id": None, "label": None, "confidence": None}

        feats = self._extract_features(frames)
        feats = self._pad_or_truncate(feats)
        x = np.expand_dims(feats, axis=0)  # (1, T, D)
        preds = self.model.predict(x, verbose=0)
        class_id = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_id])
        label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
        return {"class_id": class_id, "label": label, "confidence": confidence}

    def predict_video_bytes(self, video_bytes: bytes) -> Dict[str, float | int | str | None]:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            path = tmp.name
        try:
            return self.predict_video_path(path)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
