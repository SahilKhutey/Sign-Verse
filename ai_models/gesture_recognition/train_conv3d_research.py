"""
Research baseline trainer: clean-room Conv3D isolated sign classifier.

Note: inspired by public research directions, but implemented from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from ai_models.gesture_recognition.conv3d_dataset import Conv3DVideoDataset
from ai_models.gesture_recognition.conv3d_model import Conv3DSignModel
from training.utils.trainer_base import BaseTrainer


class Conv3DResearchTrainer(BaseTrainer):
    def build_model(self) -> nn.Module:
        num_classes = int(self.config.get("num_classes", 50))
        data_dir = self.config.get("data_dir", "training-data")
        label_map_path = os.path.join(data_dir, "video_conv3d_label_map.json")
        if os.path.exists(label_map_path):
            try:
                with open(label_map_path, "r", encoding="utf-8") as f:
                    label_to_id = json.load(f) or {}
                ids = []
                for v in label_to_id.values():
                    try:
                        ids.append(int(v))
                    except Exception:
                        continue
                if ids:
                    num_classes = max(ids) + 1
                elif label_to_id:
                    num_classes = len(label_to_id)
            except Exception:
                pass
        self.config["num_classes"] = num_classes
        return Conv3DSignModel(
            num_classes=num_classes,
            dropout=float(self.config.get("dropout", 0.3)),
        )

    def build_datasets(self):
        manifest = self.config.get(
            "manifest_csv",
            os.path.join("training-data", "video_frames_manifest.csv"),
        )
        seq_len = int(self.config.get("seq_len", 24))
        image_size = int(self.config.get("image_size", 112))
        train_ratio = float(self.config.get("train_ratio", 0.9))
        val_ratio = float(self.config.get("val_ratio", 0.1))

        train_ds = Conv3DVideoDataset(
            manifest_csv=manifest,
            seq_len=seq_len,
            image_size=image_size,
            split="train",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            augment=bool(self.config.get("augment", True)),
        )
        val_ds = Conv3DVideoDataset(
            manifest_csv=manifest,
            seq_len=seq_len,
            image_size=image_size,
            split="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            augment=False,
        )

        # Persist label map for downstream inference.
        if getattr(train_ds, "label_to_id", None):
            os.makedirs(self.config.get("data_dir", "training-data"), exist_ok=True)
            label_map_path = os.path.join(
                self.config.get("data_dir", "training-data"),
                "video_conv3d_label_map.json",
            )
            try:
                with open(label_map_path, "w", encoding="utf-8") as f:
                    json.dump(train_ds.label_to_id, f, indent=2)
            except Exception:
                pass
            self.config["num_classes"] = max(1, len(train_ds.label_to_id))

        return train_ds, val_ds

    def compute_loss(self, batch):
        x, y = batch
        logits = self.model(x)
        return F.cross_entropy(logits, y)

    def eval_metrics(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        correct1, correct5, total = 0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = self._to_device(batch)
                logits = self.model(x)
                topk = min(5, logits.size(1))
                _, topk_idx = logits.topk(topk, dim=1)
                correct1 += (logits.argmax(1) == y).sum().item()
                correct5 += topk_idx.eq(y.unsqueeze(1).expand_as(topk_idx)).any(1).sum().item()
                total += y.size(0)
        return {
            "top1_acc": correct1 / max(total, 1) * 100.0,
            "top5_acc": correct5 / max(total, 1) * 100.0,
        }


def train(config: dict):
    config.setdefault("model_name", "gesture_conv3d_research")
    trainer = Conv3DResearchTrainer(config)
    return trainer.fit()


def _load_default_config():
    try:
        import yaml
        with open("training/configs/training_config.yaml", "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("gesture_model", {}) or {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train clean-room Conv3D SLR baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--manifest-csv", default=os.path.join("training-data", "video_frames_manifest.csv"))
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--model-name", default="gesture_conv3d_research")
    args = parser.parse_args()

    cfg = _load_default_config()
    cfg["model_name"] = args.model_name
    cfg["resume"] = args.resume
    cfg["manifest_csv"] = args.manifest_csv
    cfg["seq_len"] = int(args.seq_len)
    cfg["image_size"] = int(args.image_size)
    cfg["data_dir"] = cfg.get("data_dir", "training-data")
    cfg["save_dir"] = cfg.get("save_dir", "models")
    cfg["eval_every_n_epochs"] = min(int(cfg.get("eval_every_n_epochs", 5)), 2)

    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)

    if args.dry_run:
        cfg["epochs"] = 2
        cfg["batch_size"] = min(int(cfg.get("batch_size", 4)), 4)
        cfg["seq_len"] = min(int(cfg.get("seq_len", 24)), 12)
        cfg["image_size"] = min(int(cfg.get("image_size", 112)), 96)

    train(cfg)


if __name__ == "__main__":
    main()
