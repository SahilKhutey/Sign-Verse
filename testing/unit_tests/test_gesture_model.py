"""
Unit Tests — Gesture Model
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestGestureClassifier(unittest.TestCase):

    def test_classifier_initialization(self):
        from gesture_recognition.gesture_classifier import GestureClassifier
        classifier = GestureClassifier()
        self.assertIsNotNone(classifier.gesture_map)
        self.assertEqual(len(classifier.gesture_map), 4)

    def test_empty_keypoints(self):
        from gesture_recognition.gesture_classifier import GestureClassifier
        classifier = GestureClassifier()
        result = classifier.classify([])
        self.assertIsNone(result)

    def test_valid_classification(self):
        from gesture_recognition.gesture_classifier import GestureClassifier
        classifier = GestureClassifier()
        keypoints = np.random.rand(21, 3).tolist()
        result = classifier.classify(keypoints)
        self.assertIn(result, ["HELLO", "THANK_YOU", "YES", "NO"])

    def test_gesture_model_architecture(self):
        import torch
        from gesture_recognition.models.gesture_model import GestureModel
        model = GestureModel(num_classes=4)
        dummy = torch.randn(1, 63)
        output = model(dummy)
        self.assertEqual(output.shape, (1, 4))


class TestHandTracker(unittest.TestCase):

    def test_tracker_initialization(self):
        from gesture_recognition.hand_tracking import HandTracker
        tracker = HandTracker()
        self.assertIsNotNone(tracker.hands)


class TestTemporalFilter(unittest.TestCase):

    def test_filter_smoothing(self):
        from gesture_recognition.utils.temporal_filter import TemporalFilter
        f = TemporalFilter(window=3)
        self.assertIsNone(f.update("HELLO"))
        self.assertIsNone(f.update("HELLO"))
        result = f.update("YES")
        self.assertEqual(result, "HELLO")


if __name__ == "__main__":
    unittest.main()
