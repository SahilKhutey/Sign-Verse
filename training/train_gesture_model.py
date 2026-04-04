"""
Gesture Recognition Training Entrypoint

Wrapper around `ai_models.gesture_recognition.train` so users can run:
  python training/train_gesture_model.py
  python training/train_gesture_model.py --resume
  python training/train_gesture_model.py --dry-run
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_config():
    try:
        import yaml
        with open("training/configs/training_config.yaml") as f:
            return (yaml.safe_load(f) or {}).get("gesture_model", {})
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train gesture recognition model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config()
    cfg["model_name"] = "gesture_model"
    cfg["resume"] = args.resume
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.dry_run:
        cfg["epochs"] = 2
        cfg["batch_size"] = min(int(cfg.get("batch_size", 8)), 8)

    from ai_models.gesture_recognition.train import train
    train(cfg)


if __name__ == "__main__":
    main()
