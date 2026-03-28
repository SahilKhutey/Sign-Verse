"""
Vision Pipeline — Pose Tracking Module

Full-body pose detection using MediaPipe Pose.
33 landmarks × 3 coords = 99 body features.
"""

import cv2
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
import numpy as np


class PoseTracker:

    def __init__(self, model_complexity=1, min_detection_confidence=0.5):
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp_drawing

    def process(self, frame):
        """
        Estimate body pose from frame.

        Returns:
            keypoints: numpy (99,) — body joint positions
            result: full MediaPipe result
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        keypoints = np.zeros(99)  # 33 × 3

        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                keypoints[i*3] = lm.x
                keypoints[i*3 + 1] = lm.y
                keypoints[i*3 + 2] = lm.z

        return keypoints, result

    def draw(self, frame, result):
        """Draw skeleton on frame."""
        if result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
