"""
Standalone Gesture Recognition Training Launcher

Launches gesture model training with CLI config overrides.

Usage:
    python training/launch_gesture_training.py
    python training/launch_gesture_training.py --epochs 50 --batch-size 32 --resume
    python training/launch_gesture_training.py --dry-run
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(
        description="SignVerse — Gesture Recognition Training Launcher"
    )
    parser.add_argument("--config", default="training/configs/training_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick 2-epoch validation run")
    parser.add_argument("--data-dir", type=str,
                        help="Path to training data directory")
    parser.add_argument("--num-classes", type=int,
                        help="Number of gesture classes")

    args = parser.parse_args()

    # ── Load config ──
    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f).get("gesture_model", {})
    except FileNotFoundError:
        print(f"  Config not found: {args.config}  — using defaults")
        config = {}
    except Exception as e:
        print(f"  Error reading config: {e} — using defaults")
        config = {}

    # ── Apply CLI overrides ──
    config["model_name"] = "gesture_model"
    config["resume"] = args.resume

    if args.dry_run:
        config["epochs"] = 2
        config["batch_size"] = 8
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.num_classes:
        config["num_classes"] = args.num_classes

    # ── Print config ──
    print(f"\n{'═' * 60}")
    print(f"  SignVerse — Gesture Recognition Training")
    print(f"{'═' * 60}")
    print(f"  Mode:       {'DRY-RUN' if args.dry_run else 'FULL'}")
    print(f"  Resume:     {args.resume}")
    print(f"  Epochs:     {config.get('epochs', 100)}")
    print(f"  Batch size: {config.get('batch_size', 64)}")
    print(f"  LR:         {config.get('lr', 1e-3)}")
    print(f"  Classes:    {config.get('num_classes', 2000)}")
    print(f"  Data dir:   {config.get('data_dir', 'training-data')}")
    print(f"{'─' * 60}\n")

    # ── Train ──
    t0 = time.time()

    from ai_models.gesture_recognition.train import train
    best_metric = train(config)

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  ✓ Training complete")
    print(f"  Best metric:  {best_metric:.4f}")
    print(f"  Total time:   {elapsed / 60:.1f} minutes")
    print(f"  Model saved:  models/gesture_model_best.pt")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
