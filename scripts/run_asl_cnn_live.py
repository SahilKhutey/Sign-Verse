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
    parser.add_argument("--spell-mode", action="store_true")
    parser.add_argument("--spell-window", type=int, default=8)
    parser.add_argument("--spell-majority", type=int, default=5)
    parser.add_argument("--spell-cooldown", type=int, default=4)
    parser.add_argument("--spell-blank-reset", type=int, default=3)
    args = parser.parse_args()

    classifier = ASLCNNClassifier(model_path=args.model, labels_path=args.labels)
    smoother = TemporalFilter(window=args.window)
    decoder = None
    if args.spell_mode:
        from ai_models.gesture_recognition.fingerspelling import FingerSpellingDecoder

        decoder = FingerSpellingDecoder(
            window=args.spell_window,
            min_majority=args.spell_majority,
            cooldown_frames=args.spell_cooldown,
            blank_reset_frames=args.spell_blank_reset,
            min_confidence=args.min_confidence,
        )

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
            if smooth_id is not None:
                if classifier.labels and smooth_id < len(classifier.labels):
                    label = classifier.labels[smooth_id]
                else:
                    label = str(smooth_id)

            decoded = None
            if decoder is not None:
                decoded = decoder.update(label=label, confidence=conf)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 180, 255), 2)
            live_text = f"{label or '...'} ({conf:.2f})"
            cv2.putText(frame, live_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if decoded is not None:
                stable_label = decoded.get("stable_label") or "..."
                committed = decoded.get("committed") or ""
                composed = decoded.get("text") or ""
                cv2.putText(
                    frame,
                    f"Stable: {stable_label}  Commit: {committed}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 220, 80),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Text: {composed}",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (50, 200, 255),
                    2,
                )
            cv2.imshow("ASL CNN Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and decoder is not None:
                decoder.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
