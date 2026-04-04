"""
Sequence Gesture Detection (Video)

Extracts hand keypoints from a video and produces a list of detected gestures.
"""

from __future__ import annotations

from typing import List, Dict, Any

import cv2

from gesture_recognition.hand_tracking import HandTracker
from gesture_recognition.gesture_classifier import GestureClassifier


def detect_from_video(video_path: str) -> Dict[str, Any]:
    tracker = HandTracker()
    classifier = GestureClassifier()

    cap = cv2.VideoCapture(video_path)
    gestures: List[str] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp = tracker.extract_keypoints(frame)
        if kp is None:
            continue

        g = classifier.classify([kp.tolist()])
        if g:
            gestures.append(g)

    cap.release()
    return {"gestures": gestures, "count": len(gestures)}
