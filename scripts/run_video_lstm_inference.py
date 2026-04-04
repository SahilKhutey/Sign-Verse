"""
Run isolated sign prediction on a video using the optional Keras CNN+LSTM model.

Example:
  python scripts/run_video_lstm_inference.py --video sample.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is importable when script is run directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_models.gesture_recognition.video_lstm_keras import VideoLSTMClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="models/video_lstm.h5")
    parser.add_argument("--labels", default="models/video_lstm_labels.json")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    args = parser.parse_args()

    model = VideoLSTMClassifier(
        model_path=args.model,
        labels_path=args.labels,
        max_frames=args.max_frames,
    )
    pred = model.predict_video_path(args.video)
    conf = pred.get("confidence")
    if conf is not None and float(conf) < float(args.min_confidence):
        pred["label"] = None

    print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
