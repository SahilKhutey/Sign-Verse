"""
Self-Supervised Training for Sign Language Foundation Model — Production Version

Training objective: Next gesture token prediction (GPT-style).

    Input:  G12  G34  G55  G91
    Target: G34  G55  G91  G07

Uses BaseTrainer for AMP, checkpointing, and TensorBoard logging.

Usage:
    python training/train_foundation_model.py
    python training/train_foundation_model.py --resume --dry-run
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.sign_foundation_transformer import SignFoundationModel
from training.utils.trainer_base import BaseTrainer


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

class GestureTokenDataset(Dataset):
    """Dataset of tokenized gesture sequences for self-supervised training."""

    def __init__(self, token_dir, seq_len=128):
        self.seq_len = seq_len
        self.token_dir = token_dir
        self.file_paths = []

        if os.path.exists(token_dir):
            for fname in os.listdir(token_dir):
                if fname.endswith('.npy'):
                    self.file_paths.append(os.path.join(token_dir, fname))

        if not self.file_paths:
            print(f"  No token data in {token_dir} — generating synthetic tokens")
            os.makedirs(token_dir, exist_ok=True)
            for i in range(100):
                fake_tokens = np.random.randint(1, 512, size=np.random.randint(30, 200))
                path = os.path.join(token_dir, f"seq_{i:04d}.npy")
                np.save(path, fake_tokens)
                self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        tokens = np.load(path)

        # Pad or truncate to seq_len + 1 (need input + target)
        if len(tokens) >= self.seq_len + 1:
            start = np.random.randint(0, len(tokens) - self.seq_len)
            chunk = tokens[start:start + self.seq_len + 1]
        else:
            chunk = np.pad(tokens, (0, self.seq_len + 1 - len(tokens)))

        input_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
        target_tokens = torch.tensor(chunk[1:], dtype=torch.long)

        return input_tokens, target_tokens


# ──────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────

class FoundationModelTrainer(BaseTrainer):
    """Trainer for the Sign Language Foundation Transformer."""

    def build_model(self) -> nn.Module:
        return SignFoundationModel(
            vocab_size=self.config.get("vocab_size", 512),
            d_model=self.config.get("d_model", 512),
            nhead=self.config.get("nhead", 8),
            num_layers=self.config.get("num_layers", 12),
            dim_feedforward=self.config.get("dim_feedforward", 2048),
            dropout=self.config.get("dropout", 0.1),
        )

    def build_datasets(self):
        token_dir = self.config.get("token_dir", "datasets/gesture_tokens")
        seq_len = self.config.get("seq_len", 128)

        full_ds = GestureTokenDataset(token_dir, seq_len=seq_len)
        n_total = len(full_ds)
        n_val = max(int(n_total * 0.1), 1)
        n_train = n_total - n_val

        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        return train_ds, val_ds

    def compute_loss(self, batch):
        input_tokens, target_tokens = batch
        logits = self.model(input_tokens)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=0
        )

    def eval_metrics(self):
        """Compute token accuracy and perplexity on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_correct, total_tokens, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                input_tokens, target_tokens = self._to_device(batch)
                logits = self.model(input_tokens)

                # Token accuracy (ignoring padding)
                preds = logits.argmax(-1)
                mask = target_tokens != 0
                total_correct += ((preds == target_tokens) & mask).sum().item()
                total_tokens += mask.sum().item()

                # Loss for perplexity
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                    ignore_index=0, reduction="sum"
                )
                total_loss += loss.item()

        acc = total_correct / max(total_tokens, 1) * 100
        ppl = np.exp(total_loss / max(total_tokens, 1))

        return {"token_acc": acc, "perplexity": min(ppl, 10000)}


# ──────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────

def train(config: dict):
    """Entry point called by run_all_training.py Stage 6."""
    config.setdefault("model_name", "foundation_model")
    trainer = FoundationModelTrainer(config)
    return trainer.fit()


def main():
    parser = argparse.ArgumentParser(description="Foundation Model Training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    try:
        import yaml
        with open("training/configs/training_config.yaml") as f:
            config = yaml.safe_load(f).get("foundation_model", {})
    except Exception:
        config = {}

    config["model_name"] = "foundation_model"
    config["resume"] = args.resume
    if args.dry_run:
        config["epochs"] = 2
        config["batch_size"] = 4
        print("  [DRY RUN] 2 epochs")
    if args.epochs:
        config["epochs"] = args.epochs

    train(config)


if __name__ == "__main__":
    main()
