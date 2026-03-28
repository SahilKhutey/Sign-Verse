"""
Visualization Utilities

Draw MediaPipe landmarks for debugging.
"""

from __future__ import annotations

import cv2
import mediapipe as mp


_mp_draw = mp.solutions.drawing_utils


def draw_hands(frame, results):
    if results and getattr(results, "multi_hand_landmarks", None):
        for hand in results.multi_hand_landmarks:
            _mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
    return frame


def draw_pose(frame, results):
    if results and getattr(results, "pose_landmarks", None):
        _mp_draw.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    return frame
