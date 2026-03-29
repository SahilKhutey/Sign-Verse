"""
Keras CNN ASL Alphabet Classifier

Prototype-oriented CNN path inspired by hackathon-style ASL interpreters.
Designed as an optional model path for letter-level recognition from image crops.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import cv2
import numpy as np


class ASLCNNClassifier:
    def __init__(self, model_path: str, labels_path: str | None = None, image_size: int = 224):
        self.model_path = model_path
        self.labels_path = labels_path
        self.image_size = int(image_size)

        self.model = None
        self.labels: List[str] = []
        self.input_channels = 1
        self.use_inception_preprocess = False
        self._load()

    def _load(self):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow/Keras not available. Install optional dependencies for ASL CNN."
            ) from exc

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ASL CNN model not found: {self.model_path}")
        self.model = load_model(self.model_path)

        try:
            in_shape = tuple(self.model.input_shape)  # (None, H, W, C)
            if len(in_shape) >= 4:
                if in_shape[1]:
                    self.image_size = int(in_shape[1])
                if in_shape[3]:
                    self.input_channels = int(in_shape[3])
            self.use_inception_preprocess = self.input_channels == 3 and self.image_size >= 75
        except Exception:
            pass

        if self.labels_path and os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.labels = [str(x) for x in data]
            elif isinstance(data, dict):
                self.labels = [str(v) for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")

        if self.input_channels == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            resized = cv2.resize(gray, (self.image_size, self.image_size))
            x = resized.astype(np.float32) / 255.0
            x = np.expand_dims(x, axis=-1)  # HWC, C=1
        else:
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, (self.image_size, self.image_size)).astype(np.float32)
            if self.use_inception_preprocess:
                from tensorflow.keras.applications.inception_v3 import preprocess_input  # type: ignore
                x = preprocess_input(x)
            else:
                x = x / 255.0

        x = np.expand_dims(x, axis=0)  # batch
        return x

    def predict(self, frame: np.ndarray) -> Dict[str, float | int | str | None]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        x = self.preprocess(frame)
        preds = self.model.predict(x, verbose=0)
        if preds.ndim != 2 or preds.shape[0] == 0:
            return {"class_id": None, "label": None, "confidence": None}
        class_id = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_id])
        label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
        return {"class_id": class_id, "label": label, "confidence": confidence}
