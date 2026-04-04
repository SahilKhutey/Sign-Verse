"""
Realtime Gesture Detection (Webcam)

Baseline pipeline:
  webcam frame -> hand keypoints -> placeholder classifier -> label overlay
"""

from __future__ import annotations

import cv2

from gesture_recognition.hand_tracking import HandTracker
from gesture_recognition.gesture_classifier import GestureClassifier


def run(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    tracker = HandTracker()
    classifier = GestureClassifier()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp = tracker.extract_keypoints(frame)
        label = classifier.classify([kp.tolist()]) if kp is not None else None

        if label:
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("SignVerse Realtime Gesture", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
