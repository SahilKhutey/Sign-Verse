"""
Research baseline trainer: clean-room Pose-GCN style classifier.

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
from torch.utils.data import Subset

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from ai_models.gesture_recognition.dataset import GestureDataset
from ai_models.gesture_recognition.pose_gcn_model import PoseGCNSignModel
from training.utils.trainer_base import BaseTrainer


class PoseGCNResearchTrainer(BaseTrainer):
    def build_model(self) -> nn.Module:
        num_classes = int(self.config.get("num_classes", 2000))
        data_dir = self.config.get("data_dir", "training-data")
        label_map_path = os.path.join(data_dir, "label_map.json")
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

        feature_dim = int(self.config.get("feature_dim", 225))
        num_nodes = max(1, feature_dim // 3)
        return PoseGCNSignModel(
            num_classes=num_classes,
            num_nodes=num_nodes,
            dropout=float(self.config.get("dropout", 0.2)),
        )

    def build_datasets(self):
        data_dir = self.config.get("data_dir", "training-data")
        seq_len = int(self.config.get("seq_len", 60))
        feature_dim = int(self.config.get("feature_dim", 225))
        augment = bool(self.config.get("augment", True))

        full_ds = GestureDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            feature_dim=feature_dim,
            augment=augment,
        )
        max_samples = int(self.config.get("max_samples", 0))
        if max_samples > 0 and len(full_ds) > max_samples:
            idx = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(42))
            full_ds = Subset(full_ds, idx[:max_samples].tolist())

        n_total = len(full_ds)
        if n_total <= 1:
            # GestureDataset already creates synthetic sample on empty dataset.
            return full_ds, None

        n_val = max(1, int(round(n_total * float(self.config.get("val_ratio", 0.1)))))
        n_val = min(n_val, n_total - 1)
        n_train = n_total - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
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
    config.setdefault("model_name", "gesture_pose_gcn_research")
    trainer = PoseGCNResearchTrainer(config)
    return trainer.fit()


def _load_default_config():
    try:
        import yaml
        with open("training/configs/training_config.yaml", "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("gesture_model", {}) or {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train clean-room Pose-GCN SLR baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--feature-dim", type=int, default=225)
    parser.add_argument("--model-name", default="gesture_pose_gcn_research")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_default_config()
    cfg["model_name"] = args.model_name
    cfg["resume"] = args.resume
    cfg["seq_len"] = int(args.seq_len)
    cfg["feature_dim"] = int(args.feature_dim)
    cfg["max_samples"] = int(args.max_samples)
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
        cfg["batch_size"] = min(int(cfg.get("batch_size", 16)), 16)
        cfg["seq_len"] = min(int(cfg.get("seq_len", 60)), 12)
        cfg["max_samples"] = min(
            int(cfg.get("max_samples", 64)) if int(cfg.get("max_samples", 0)) > 0 else 64,
            64,
        )

    train(cfg)


if __name__ == "__main__":
    main()
