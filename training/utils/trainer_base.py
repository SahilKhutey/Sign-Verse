"""
BaseTrainer — Reusable training base class for all SignVerse models.

Features:
    - Automatic Mixed Precision (AMP) via torch.cuda.amp.GradScaler
    - Checkpoint save / load / resume via CheckpointManager
    - TensorBoard / CSV logging via TensorBoardLogger
    - Weights & Biases (WandB) integration
    - Multi-GPU support via DataParallel
    - Gradient clipping
    - Cosine annealing LR schedule
    - Train + optional validation loop

Subclass and override:
    - build_model()     → return nn.Module
    - build_datasets()  → return (train_dataset, val_dataset or None)
    - loss_fn()         → return scalar tensor loss
    - eval_metrics()    → return dict[str,float]  (optional)
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from training.utils.checkpoint_manager import CheckpointManager
from training.utils.tensorboard_logger import TensorBoardLogger


class BaseTrainer:
    """
    Base trainer class with AMP, checkpointing, and logging.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._resolve_device(config.get("device", "auto"))
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 32)
        self.lr = config.get("lr", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.save_dir = config.get("save_dir", "models")
        self.log_dir = config.get("log_dir", os.path.join("logs", "tensorboard"))
        self.num_workers = config.get("num_workers", 0)
        self.save_every = config.get("save_every_n_epochs", 10)
        self.eval_every = config.get("eval_every_n_epochs", 5)
        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.model_name = config.get("model_name", "model")
        self.resume = config.get("resume", False)

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Will be initialized in .setup()
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.logger: Optional[TensorBoardLogger] = None
        self.wandb_logger = None
        self.ckpt_manager: Optional[CheckpointManager] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.best_metric = float("inf")
        self.start_epoch = 0
        self.global_step = 0

    # ──────────────────────────────────────────────────────────────
    # Abstract / overridable
    # ──────────────────────────────────────────────────────────────

    def build_model(self) -> nn.Module:
        raise NotImplementedError("Subclass must implement build_model()")

    def build_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Return (train_dataset, val_dataset). val_dataset may be None."""
        raise NotImplementedError("Subclass must implement build_datasets()")

    def compute_loss(self, batch) -> torch.Tensor:
        """Given a batch, return scalar loss tensor."""
        raise NotImplementedError("Subclass must implement compute_loss()")

    def eval_metrics(self) -> Dict[str, float]:
        """Override to add validation-time metrics (accuracy, BLEU, etc.)."""
        return {}

    # ──────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────

    def setup(self):
        """Initialize model, optimizer, scheduler, AMP, logging, dataloaders."""
        # Model
        self.model = self.build_model()
        
        # Multi-GPU support
        gpu_count = self.config.get("gpus", 1)
        if self.device.type == "cuda" and gpu_count > 1:
            n_available = torch.cuda.device_count()
            if n_available < gpu_count:
                print(f"  Warning: requested {gpu_count} GPUs, but only {n_available} available.")
                gpu_count = n_available
            if gpu_count > 1:
                print(f"  Using {gpu_count} GPUs via DataParallel")
                self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [{self.model_name}] Parameters: {n_params:,} | Device: {self.device} | AMP: {self.use_amp}")

        # Optimiser + scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        # AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logging
        self.logger = TensorBoardLogger(
            log_dir=os.path.join(self.log_dir, self.model_name)
        )
        
        # WandB logging
        if self.config.get("monitor") == "wandb":
            try:
                import wandb
                print(f"  Initializing Weights & Biases for {self.model_name}...")
                wandb.init(
                    project="SignVerse",
                    name=self.model_name,
                    config=self.config,
                    reinit=True
                )
                self.wandb_logger = wandb
            except ImportError:
                print("  Warning: 'wandb' package not found. Skipping WandB logging.")

        # Checkpointing
        self.ckpt_manager = CheckpointManager(
            save_dir=self.save_dir,
            prefix=self.model_name
        )

        # Resume?
        if self.resume:
            ckpt = self.ckpt_manager.find_latest()
            if ckpt:
                self.start_epoch, self.best_metric = self.ckpt_manager.load(
                    ckpt, self.model, self.optimizer
                )
                # Fast-forward scheduler
                for _ in range(self.start_epoch):
                    self.scheduler.step()
                print(f"  Resumed from epoch {self.start_epoch} | best: {self.best_metric:.4f}")

        # Datasets
        train_ds, val_ds = self.build_datasets()
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda")
        )
        if val_ds is not None:
            self.val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self.device.type == "cuda")
            )
        print(f"  Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader) if self.val_loader else 0}")

    # ──────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward + loss with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.compute_loss(batch)

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            self.global_step += 1

            # Log every ~10% of epoch
            log_interval = max(1, n_batches // 10)
            if (batch_idx + 1) % log_interval == 0:
                self.logger.log_metrics(
                    {"train/loss_step": loss.item()},
                    step=self.global_step
                )

        avg_loss = total_loss / max(n_batches, 1)
        log_metrics = {"train/loss": avg_loss, "train/lr": self.scheduler.get_last_lr()[0]}
        
        self.logger.log_metrics(log_metrics, step=epoch)
        if self.wandb_logger:
            self.wandb_logger.log(log_metrics, step=epoch)
        
        return avg_loss

    def val_epoch(self, epoch: int) -> float:
        """Run validation. Returns average val loss."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self.compute_loss(batch)
                total_loss += loss.item()

        avg_loss = total_loss / max(len(self.val_loader), 1)

        # Extra metrics (accuracy, BLEU, etc.)
        extra = self.eval_metrics()
        log_dict = {"val/loss": avg_loss, **{f"val/{k}": v for k, v in extra.items()}}
        
        self.logger.log_metrics(log_dict, step=epoch)
        if self.wandb_logger:
            self.wandb_logger.log(log_dict, step=epoch)
            
        return avg_loss

    def fit(self):
        """Full training loop."""
        self.setup()

        print(f"\n  Starting training for {self.epochs} epochs (from {self.start_epoch})...")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()

            train_loss = self.train_epoch(epoch)
            self.scheduler.step()

            val_loss = float("inf")
            if (epoch + 1) % self.eval_every == 0 and self.val_loader:
                val_loss = self.val_epoch(epoch)

            elapsed = time.time() - t0
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f" | Val: {val_loss:.4f}" if val_loss != float("inf") else ""
                print(f"  Epoch {epoch+1:4d}/{self.epochs} | Loss: {train_loss:.4f}{val_str} | {elapsed:.1f}s")

            # Save periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.ckpt_manager.save(
                    self.model, self.optimizer, epoch + 1, train_loss,
                    tag=f"epoch{epoch+1}"
                )

            # Save best
            metric = val_loss if val_loss != float("inf") else train_loss
            if metric < self.best_metric:
                self.best_metric = metric
                self.ckpt_manager.save(
                    self.model, self.optimizer, epoch + 1, metric,
                    tag="best"
                )

        # ASCII-only to avoid Windows console encoding crashes.
        print(f"\n  Training complete. Best metric: {self.best_metric:.4f}")
        self.logger.close()
        if self.wandb_logger:
            self.wandb_logger.finish()
        return self.best_metric

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(setting: str) -> torch.device:
        if setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(setting)

    def _to_device(self, batch):
        """Recursively move a batch (tuple/list/tensor/dict) to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch
