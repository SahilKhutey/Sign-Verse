"""
Live Capture Pipeline (OpenCV)

Captures frames from webcam, extracts keypoints, predicts gesture IDs,
and overlays results in real time.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from api_server.model_loader import ModelLoader
from api_server.realtime_inference import RealtimeInference
from vision_pipeline.feature_extractor import FeatureExtractor
from gesture_recognition.utils.temporal_filter import TemporalFilter


class LiveCapture:
    def __init__(
        self,
        camera_index: int = 0,
        window_name: str = "SignVerse Live",
        window: int = 8,
        fps_limit: int = 15,
    ):
        self.camera_index = camera_index
        self.window_name = window_name
        self.fps_limit = fps_limit
        self.filter = TemporalFilter(window=window)
        self.loader = ModelLoader()
        self.realtime = RealtimeInference(self.loader)
        self.extractor = FeatureExtractor()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")

        prev_time = 0.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.time()
                if self.fps_limit > 0 and (now - prev_time) < (1 / self.fps_limit):
                    continue
                prev_time = now

                features, drawn = self.extractor.extract_and_draw(frame)
                result = self.realtime.classify_gesture(features.tolist())
                gesture_id = result.get("gesture_id")
                smoothed = self.filter.update(gesture_id)

                label = None
                if smoothed is not None:
                    label = self.loader.gesture_label(smoothed) or f"ID:{smoothed}"

                # Overlay text
                overlay_text = label or "Detecting..."
                cv2.putText(
                    drawn,
                    overlay_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(self.window_name, drawn)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def run_live(camera_index: int = 0):
    LiveCapture(camera_index=camera_index).run()
