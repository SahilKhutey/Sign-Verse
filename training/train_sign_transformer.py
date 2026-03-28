"""
Sign->Text Transformer Training Entrypoint

Wrapper around `ai_models.sign_transformer.train_transformer` so users can run:
  python training/train_sign_transformer.py
  python training/train_sign_transformer.py --resume
  python training/train_sign_transformer.py --dry-run
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_config():
    try:
        import yaml
        with open("training/configs/training_config.yaml") as f:
            return (yaml.safe_load(f) or {}).get("sign_transformer", {})
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train sign->text transformer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config()
    cfg["model_name"] = "sign_transformer"
    cfg["resume"] = args.resume
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.dry_run:
        cfg["epochs"] = 2
        cfg["batch_size"] = min(int(cfg.get("batch_size", 4)), 4)

    from ai_models.sign_transformer.train_transformer import train
    train(cfg)


if __name__ == "__main__":
    main()
