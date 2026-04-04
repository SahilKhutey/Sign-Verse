"""
Vision Pipeline — Hand Tracking Module

Real-time hand detection and keypoint extraction.
Uses MediaPipe Hands for 21-landmark per-hand tracking.
"""

import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import numpy as np


class HandTracker:
    """Tracks up to 2 hands — 21 landmarks × 3 coords × 2 = 126 features."""

    def __init__(self, max_hands=2, detection_confidence=0.6,
                 tracking_confidence=0.6, model_complexity=0):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=model_complexity
        )
        self.mp_draw = mp_drawing
        self.max_hands = max_hands

    def process(self, frame):
        """
        Detect hands in frame.

        Returns:
            keypoints: numpy (126,) — zeros if no hands detected
            result: full MediaPipe result for visualization
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        keypoints = np.zeros(self.max_hands * 63)

        if result.multi_hand_landmarks:
            for i, hand in enumerate(result.multi_hand_landmarks[:self.max_hands]):
                offset = i * 63
                for j, lm in enumerate(hand.landmark):
                    keypoints[offset + j*3] = lm.x
                    keypoints[offset + j*3 + 1] = lm.y
                    keypoints[offset + j*3 + 2] = lm.z

        return keypoints, result

    def draw(self, frame, result):
        """Draw hand landmarks on frame."""
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS
                )
        return frame
