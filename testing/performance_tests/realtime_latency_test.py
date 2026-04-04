"""
Performance Tests — Real-time latency benchmarks.

Measures processing time for each pipeline component
to verify real-time performance targets.

Targets:
    Camera capture:    ~3 ms
    Hand tracking:     ~10 ms
    Gesture inference: ~5 ms
    Processing:        ~2 ms
    Total:             ~20 ms per frame → ~50 FPS
"""

import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestRealtimeLatency(unittest.TestCase):

    def test_text_to_sign_latency(self):
        from ai_engine.modules.text_to_sign import TextToSignConverter
        converter = TextToSignConverter()

        start = time.perf_counter()
        for _ in range(100):
            converter.convert("I am going to school tomorrow")
        elapsed = (time.perf_counter() - start) / 100

        print(f"Text-to-sign latency: {elapsed*1000:.2f} ms")
        self.assertLess(elapsed, 0.01)  # <10ms

    def test_gesture_classifier_latency(self):
        from gesture_recognition.gesture_classifier import GestureClassifier
        classifier = GestureClassifier()

        keypoints = np.random.rand(21, 3).tolist()

        start = time.perf_counter()
        for _ in range(100):
            classifier.classify(keypoints)
        elapsed = (time.perf_counter() - start) / 100

        print(f"Gesture classifier latency: {elapsed*1000:.2f} ms")
        self.assertLess(elapsed, 0.01)  # <10ms

    def test_temporal_filter_latency(self):
        from gesture_recognition.utils.temporal_filter import TemporalFilter
        f = TemporalFilter(window=10)

        start = time.perf_counter()
        for _ in range(1000):
            f.update("HELLO")
        elapsed = (time.perf_counter() - start) / 1000

        print(f"Temporal filter latency: {elapsed*1000:.4f} ms")
        self.assertLess(elapsed, 0.001)  # <1ms

    def test_gesture_model_inference_latency(self):
        import torch
        from gesture_recognition.models.gesture_model import GestureModel

        model = GestureModel(num_classes=4)
        model.eval()
        dummy = torch.randn(1, 63)

        # Warmup
        for _ in range(10):
            model(dummy)

        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                model(dummy)
        elapsed = (time.perf_counter() - start) / 100

        print(f"Gesture model inference: {elapsed*1000:.2f} ms")
        self.assertLess(elapsed, 0.01)  # <10ms

    def test_animation_mapper_latency(self):
        from avatar_animation.gesture_mapper import GestureMapper
        mapper = GestureMapper()

        tokens = ["HELLO", "THANK_YOU", "YES", "NO", "I"]

        start = time.perf_counter()
        for _ in range(100):
            mapper.map_sequence(tokens)
        elapsed = (time.perf_counter() - start) / 100

        print(f"Animation mapper latency: {elapsed*1000:.4f} ms")
        self.assertLess(elapsed, 0.005)  # <5ms


if __name__ == "__main__":
    unittest.main(verbosity=2)
