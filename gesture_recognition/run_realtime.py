"""
Real-Time Sign Language Gesture Recognition Pipeline

Pipeline:
    Camera Frame → Hand Detection (MediaPipe) → Keypoint Extraction (21 points)
    → Feature Vector (63 values) → Gesture Classification Model
    → Temporal Filter → Recognized Gesture

Target latency: ~20ms per frame (~50 FPS possible)

    Component          Time
    Camera capture     ~3 ms
    Hand tracking      ~10 ms
    Gesture inference  ~5 ms
    Processing         ~2 ms

Optimizations:
    1. MediaPipe GPU if available
    2. Camera resolution 640×480 (not 1080p)
    3. Frame skipping (process every 2nd frame) if needed
    4. TorchScript model export for faster inference

Expected Performance:
    Laptop CPU:   25–35 FPS
    Laptop GPU:   50–80 FPS
    Mobile phone: 20–30 FPS
"""

import cv2
from hand_tracking import HandTracker
from realtime_gesture import RealtimeGestureRecognizer
from utils.temporal_filter import TemporalFilter


labels = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO"
]

tracker = HandTracker()

recognizer = RealtimeGestureRecognizer(
    "models/gesture_model/gesture_classifier.pt",
    labels
)

filter = TemporalFilter()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    keypoints = tracker.extract_keypoints(frame)

    gesture = recognizer.predict(keypoints)

    smoothed = filter.update(gesture)

    if smoothed:
        cv2.putText(
            frame,
            smoothed,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
