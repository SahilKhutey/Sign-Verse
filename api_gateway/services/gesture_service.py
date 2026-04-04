"""
Gesture Service — Backend logic for gesture recognition.
"""

import numpy as np


class GestureServiceHandler:

    @staticmethod
    def classify(keypoints):
        from gesture_recognition.gesture_classifier import GestureClassifier
        classifier = GestureClassifier()
        gesture = classifier.classify([keypoints])
        return {"gesture": gesture}

    @staticmethod
    def detect_from_video(video_path):
        import cv2
        from gesture_recognition.hand_tracking import HandTracker
        from gesture_recognition.gesture_classifier import GestureClassifier

        tracker = HandTracker()
        classifier = GestureClassifier()

        cap = cv2.VideoCapture(video_path)
        gestures = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = tracker.extract_keypoints(frame)
            if keypoints is not None:
                gesture = classifier.classify([keypoints.tolist()])
                if gesture:
                    gestures.append(gesture)

        cap.release()
        return {"gestures": gestures, "count": len(gestures)}
