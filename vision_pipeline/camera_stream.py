"""
Camera Stream — Manages video capture and frame distribution.

Provides a unified interface for:
    - Webcam input
    - Video file input
    - AR device stream
"""

import cv2
import threading
import queue
import time


class CameraStream:
    """Thread-safe camera stream with frame queue."""

    def __init__(self, source=0, width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self._thread = None
        self.cap = None

    def start(self):
        """Start the camera capture thread."""
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"Camera stream started: {self.width}x{self.height} @ {self.fps}fps")

    def _capture_loop(self):
        """Continuously capture frames into queue."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str):
                    # Video file ended — rewind
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Drop oldest frame if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)
            time.sleep(1.0 / self.fps)

    def get_frame(self, timeout=0.1):
        """Get the latest frame from the queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop the camera stream."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print("Camera stream stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
