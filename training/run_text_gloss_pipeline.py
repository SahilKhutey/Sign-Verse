"""
End-to-end text<->gloss pipeline:
1) Build dataset splits
2) Train text<->gloss models

Usage:
  python training/run_text_gloss_pipeline.py --curriculum --augment --register
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    build_cmd = [
        sys.executable,
        os.path.join("training", "data_pipeline", "build_text_gloss_dataset.py"),
        "--min-len", str(args.min_len),
        "--max-len", str(args.max_len),
    ]
    if args.max_pairs:
        build_cmd += ["--max-pairs", str(args.max_pairs)]

    train_cmd = [
        sys.executable,
        os.path.join("training", "train_text_gloss.py"),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
    ]
    if args.curriculum:
        train_cmd.append("--curriculum")
    if args.augment:
        train_cmd.append("--augment")
    if args.register:
        train_cmd.append("--register")

    print("==> Building text<->gloss dataset")
    subprocess.check_call(build_cmd)
    print("==> Training text<->gloss models")
    subprocess.check_call(train_cmd)


if __name__ == "__main__":
    main()
