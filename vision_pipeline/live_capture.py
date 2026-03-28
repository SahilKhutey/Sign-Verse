"""
Live Capture Pipeline (OpenCV)

Captures frames from webcam, extracts keypoints, predicts gesture IDs,
and overlays results in real time.
"""

from __future__ import annotations

import time
from typing import Optional, List

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
        width: Optional[int] = None,
        height: Optional[int] = None,
        flip: bool = False,
        show_fps: bool = True,
        draw_guides: bool = True,
        min_confidence: float = 0.4,
        tts: bool = False,
        tts_cooldown: float = 2.0,
    ):
        self.camera_index = camera_index
        self.window_name = window_name
        self.fps_limit = fps_limit
        self.width = width
        self.height = height
        self.flip = flip
        self.show_fps = show_fps
        self.draw_guides = draw_guides
        self.min_confidence = min_confidence
        self.filter = TemporalFilter(window=window)
        self.loader = ModelLoader()
        self.realtime = RealtimeInference(self.loader)
        self.extractor = FeatureExtractor()
        self.tts_enabled = tts
        self.tts_cooldown = tts_cooldown
        self._tts = None
        self._last_spoken = 0.0

    def _apply_camera_settings(self, cap: cv2.VideoCapture) -> None:
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def _draw_guides(self, frame):
        h, w = frame.shape[:2]
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.12)
        color = (90, 90, 90)
        cv2.rectangle(
            frame,
            (margin_x, margin_y),
            (w - margin_x, h - margin_y),
            color,
            1,
        )
        cv2.line(frame, (w // 2, margin_y), (w // 2, h - margin_y), color, 1)
        cv2.line(frame, (margin_x, h // 2), (w - margin_x, h // 2), color, 1)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        self._apply_camera_settings(cap)

        prev_time = 0.0
        fps_smoothed = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.flip:
                    frame = cv2.flip(frame, 1)

                now = time.time()
                if self.fps_limit > 0 and (now - prev_time) < (1 / self.fps_limit):
                    continue
                if prev_time > 0:
                    instant_fps = 1.0 / max(now - prev_time, 1e-6)
                    fps_smoothed = instant_fps if fps_smoothed is None else (0.85 * fps_smoothed + 0.15 * instant_fps)
                prev_time = now

                features, drawn = self.extractor.extract_and_draw(frame)
                result = self.realtime.classify_gesture(features.tolist())
                gesture_id = result.get("gesture_id")
                confidence = result.get("confidence")
                if confidence is not None and confidence < self.min_confidence:
                    gesture_id = None
                smoothed = self.filter.update(gesture_id)

                label = None
                if smoothed is not None:
                    label = self.loader.gesture_label(smoothed) or f"ID:{smoothed}"

                if self.draw_guides:
                    self._draw_guides(drawn)

                # Overlay text
                overlay_text = label or "Detecting..."
                if confidence is not None:
                    overlay_text = f"{overlay_text} ({confidence:.2f})"
                cv2.putText(
                    drawn,
                    overlay_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                if self.show_fps and fps_smoothed is not None:
                    cv2.putText(
                        drawn,
                        f"{fps_smoothed:.1f} FPS",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (120, 200, 255),
                        2,
                    )

                if self.tts_enabled and label:
                    now = time.time()
                    if (now - self._last_spoken) >= self.tts_cooldown:
                        try:
                            if self._tts is None:
                                from ai_engine.modules.text_to_speech import TextToSpeech
                                self._tts = TextToSpeech()
                            self._tts.speak(label)
                            self._last_spoken = now
                        except Exception:
                            pass

                cv2.imshow(self.window_name, drawn)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

def list_cameras(max_index: int = 6) -> List[int]:
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, _ = cap.read()
        cap.release()
        if ret:
            available.append(idx)
    return available


def run_live(
    camera_index: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps_limit: int = 15,
    window: int = 8,
    flip: bool = False,
    show_fps: bool = True,
    draw_guides: bool = True,
    min_confidence: float = 0.4,
    tts: bool = False,
    tts_cooldown: float = 2.0,
):
    LiveCapture(
        camera_index=camera_index,
        width=width,
        height=height,
        fps_limit=fps_limit,
        window=window,
        flip=flip,
        show_fps=show_fps,
        draw_guides=draw_guides,
        min_confidence=min_confidence,
        tts=tts,
        tts_cooldown=tts_cooldown,
    ).run()
