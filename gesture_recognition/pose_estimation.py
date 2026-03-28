"""
Pose Estimation (Body)

Extracts 33 MediaPipe Pose landmarks (x,y,z) as a flat 99-dim vector.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def extract_keypoints(self, frame) -> np.ndarray:
        """Return a (99,) vector. Zeros if pose not detected."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        keypoints = np.zeros(99, dtype=np.float32)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                keypoints[i * 3] = lm.x
                keypoints[i * 3 + 1] = lm.y
                keypoints[i * 3 + 2] = lm.z
        return keypoints
