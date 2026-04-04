"""
Gesture Diffusion Training — Production Version

Trains the gesture diffusion model to denoise motion vectors.

Objective:
    Learn to reconstruct real gesture motion from noise.
    noise + timestep → model → predicted_noise
    loss = MSE(predicted_noise, actual_noise)

Uses BaseTrainer for AMP, checkpointing, and TensorBoard logging.

Usage:
    python training/train_diffusion.py
    python training/train_diffusion.py --resume --dry-run
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

from models.gesture_diffusion import GestureDiffusionModel, DiffusionScheduler
from training.utils.trainer_base import BaseTrainer


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

class MotionDataset(Dataset):
    """Dataset of motion sequences for diffusion training."""

    def __init__(self, data_dir, tokens_dir=None, seq_len=30, motion_dim=150, token_len=64):
        self.seq_len = seq_len
        self.motion_dim = motion_dim
        self.token_len = token_len
        self.sequences = []
        self.tokens = []

        if tokens_dir is None:
            tokens_dir = os.path.join(os.path.dirname(data_dir), "gesture_tokens")

        if os.path.exists(data_dir):
            for fname in sorted(os.listdir(data_dir)):
                if fname.endswith('.npy'):
                    # Load motion
                    data = np.load(os.path.join(data_dir, fname))
                    self.sequences.append(data)
                    
                    # Try to load matching tokens
                    # e.g. motion_0000.npy -> sample_000000_tokens.npy
                    idx_str = "".join(filter(str.isdigit, fname))
                    token_fname = f"sample_{int(idx_str):06d}_tokens.npy"
                    token_path = os.path.join(tokens_dir, token_fname)
                    
                    if os.path.exists(token_path):
                        self.tokens.append(np.load(token_path))
                    else:
                        # Fallback to random tokens if not found
                        self.tokens.append(np.random.randint(0, 512, (token_len,)).astype(np.int64))

        if not self.sequences:
            print(f"  No motion data in {data_dir} — generating synthetic data")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(tokens_dir, exist_ok=True)
            for i in range(50):
                fake_motion = np.random.randn(
                    np.random.randint(30, 120), motion_dim
                ).astype(np.float32) * 0.1
                np.save(os.path.join(data_dir, f"motion_{i:04d}.npy"), fake_motion)
                self.sequences.append(fake_motion)
                
                fake_tokens = np.random.randint(0, 512, (token_len,)).astype(np.int64)
                np.save(os.path.join(tokens_dir, f"sample_{i:06d}_tokens.npy"), fake_tokens)
                self.tokens.append(fake_tokens)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx % len(self.sequences)]
        tokens = self.tokens[idx % len(self.tokens)]

        if len(seq) >= self.seq_len:
            start = np.random.randint(0, len(seq) - self.seq_len + 1)
            chunk = seq[start:start + self.seq_len]
        else:
            chunk = np.pad(seq, ((0, self.seq_len - len(seq)), (0, 0)))

        # Ensure correct motion dimension
        if chunk.shape[-1] > self.motion_dim:
            chunk = chunk[:, :self.motion_dim]
        elif chunk.shape[-1] < self.motion_dim:
            pad_width = self.motion_dim - chunk.shape[-1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_width)))

        # Handle tokens padding/truncation
        if len(tokens) > self.token_len:
            tokens = tokens[:self.token_len]
        elif len(tokens) < self.token_len:
            tokens = np.pad(tokens, (0, self.token_len - len(tokens)))

        return torch.tensor(chunk, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────

class DiffusionTrainer(BaseTrainer):
    """
    Trainer for the Gesture Diffusion Model.

    Unlike classification models, diffusion training:
    1. Samples random timesteps per batch
    2. Adds noise to the motion at those timesteps
    3. Model predicts the noise
    4. Loss = MSE(predicted_noise, actual_noise)
    """

    def __init__(self, config):
        super().__init__(config)
        self.diffusion_scheduler = DiffusionScheduler(
            num_timesteps=config.get("num_timesteps", 1000)
        )

    def build_model(self) -> nn.Module:
        return GestureDiffusionModel(
            motion_dim=self.config.get("motion_dim", 150),
            d_model=self.config.get("d_model", 512),
            num_blocks=self.config.get("num_blocks", 6),
        )

    def build_datasets(self):
        data_dir = self.config.get("data_dir", "datasets/pose_keypoints")
        tokens_dir = self.config.get("tokens_dir", "datasets/gesture_tokens")
        seq_len = self.config.get("seq_len", 30)
        motion_dim = self.config.get("motion_dim", 150)
        token_len = self.config.get("token_len", 64)

        full_ds = MotionDataset(data_dir, tokens_dir=tokens_dir, seq_len=seq_len, motion_dim=motion_dim, token_len=token_len)
        n_total = len(full_ds)
        n_val = max(int(n_total * 0.1), 1)
        n_train = n_total - n_val

        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        return train_ds, val_ds

    def compute_loss(self, batch):
        """
        Diffusion-specific loss:
        1. Sample random timesteps
        2. Add noise
        3. Predict noise
        4. MSE loss
        """
        # batch is (motion_batch, tokens_batch)
        motion_batch, tokens_batch = batch
        motion_batch = motion_batch.to(self.device)
        tokens_batch = tokens_batch.to(self.device)

        # Sample random timesteps
        bs = motion_batch.size(0)
        t = self.diffusion_scheduler.sample_timestep(bs).to(self.device)

        # Add noise at timestep t
        noisy_motion, noise = self.diffusion_scheduler.add_noise(motion_batch, t)
        noisy_motion = noisy_motion.to(self.device)
        noise = noise.to(self.device)

        # Model predicts the noise, now with gesture tokens for conditioning
        predicted_noise = self.model(noisy_motion, t, gesture_tokens=tokens_batch)

        # MSE loss between predicted and actual noise
        return F.mse_loss(predicted_noise, noise)


# ──────────────────────────────────────────────────────────────────
# Motion generation utility
# ──────────────────────────────────────────────────────────────────

def generate_motion(model_path, num_samples=1, seq_len=30, motion_dim=150):
    """
    Generate gesture motion from trained diffusion model.

    Returns: (num_samples, seq_len, motion_dim) tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GestureDiffusionModel(motion_dim=motion_dim).to(device)
    payload = torch.load(model_path, map_location=device)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)

    scheduler = DiffusionScheduler()

    motion = scheduler.generate(
        model,
        shape=(num_samples, seq_len, motion_dim),
        device=device
    )

    print(f"Generated {num_samples} motion sequences of {seq_len} frames")
    return motion


# ──────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────

def train(config: dict):
    """Entry point called by run_all_training.py Stage 8."""
    config.setdefault("model_name", "diffusion_model")
    trainer = DiffusionTrainer(config)
    return trainer.fit()


def main():
    parser = argparse.ArgumentParser(description="Gesture Diffusion Training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    try:
        import yaml
        with open("training/configs/training_config.yaml") as f:
            config = yaml.safe_load(f).get("diffusion_model", {})
    except Exception:
        config = {}

    config["model_name"] = "diffusion_model"
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
