"""
Run live ASL CNN inference from webcam.

Usage:
  python scripts/run_asl_cnn_live.py --camera 0
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

# Ensure project root is importable when script is run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_models.gesture_recognition.cnn_asl import ASLCNNClassifier
from gesture_recognition.utils.temporal_filter import TemporalFilter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="models/asl_cnn.h5")
    parser.add_argument("--labels", default="models/asl_cnn_labels.json")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--window", type=int, default=6)
    args = parser.parse_args()

    classifier = ASLCNNClassifier(model_path=args.model, labels_path=args.labels)
    smoother = TemporalFilter(window=args.window)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            size = min(h, w) // 2
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            x2 = x1 + size
            y2 = y1 + size
            roi = frame[y1:y2, x1:x2]

            pred = classifier.predict(roi)
            class_id = pred.get("class_id")
            conf = pred.get("confidence") or 0.0
            if conf < args.min_confidence:
                class_id = None

            smooth_id = smoother.update(class_id)
            label = pred.get("label")
            if smooth_id is not None and classifier.labels and smooth_id < len(classifier.labels):
                label = classifier.labels[smooth_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 180, 255), 2)
            text = f"{label or '...'} ({conf:.2f})"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("ASL CNN Live", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
