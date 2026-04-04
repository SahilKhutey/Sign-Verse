"""
Train sign->text transformer on video-derived frame-feature sequences.

Typical flow:
1) Build dataset:
   python training/data_pipeline/build_video_sign_text_dataset.py
2) Train:
   python training/train_video_sign_to_text.py --epochs 12
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_config():
    try:
        import yaml
        with open("training/configs/training_config.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("sign_transformer", {}) or {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train video sign->text transformer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--labels-csv", default=os.path.join("training-data", "video_sign_text_labels.csv"))
    parser.add_argument("--keypoint-dir", default=os.path.join("training-data", "video_keypoints"))
    parser.add_argument("--model-name", default="video_sign_transformer")
    parser.add_argument("--vocab-path", default=os.path.join("models", "video_sign_transformer_vocab.json"))
    parser.add_argument("--feature-dim", type=int, default=225)
    parser.add_argument("--src-len", type=int, default=90)
    parser.add_argument("--tgt-len", type=int, default=30)
    args = parser.parse_args()

    cfg = _load_config()
    cfg["model_name"] = args.model_name
    cfg["resume"] = args.resume
    cfg["labels_csv"] = args.labels_csv
    cfg["keypoint_dir"] = args.keypoint_dir
    cfg["vocab_path"] = args.vocab_path
    cfg["feature_dim"] = int(args.feature_dim)
    cfg["src_len"] = int(args.src_len)
    cfg["tgt_len"] = int(args.tgt_len)

    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["learning_rate"] = float(args.lr)

    if args.dry_run:
        cfg["epochs"] = 2
        cfg["batch_size"] = min(int(cfg.get("batch_size", 4)), 4)
        cfg["d_model"] = 128
        cfg["nhead"] = 4
        cfg["num_layers"] = 2
        cfg["src_len"] = min(int(cfg.get("src_len", 90)), 30)
        cfg["tgt_len"] = min(int(cfg.get("tgt_len", 30)), 12)

    from ai_models.sign_transformer.train_transformer import train

    train(cfg)


if __name__ == "__main__":
    main()
