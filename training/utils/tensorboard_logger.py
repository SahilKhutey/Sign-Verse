"""
TensorBoardLogger — Unified logging for SignVerse training.

Primary:  torch.utils.tensorboard.SummaryWriter  (if available)
Fallback: CSV file  (always works, no extra dependencies)

Usage:
    logger = TensorBoardLogger(log_dir="logs/tensorboard/gesture_model")
    logger.log_metrics({"train/loss": 0.42, "train/acc": 87.3}, step=10)
    logger.log_lr(0.001, step=10)
    logger.close()
"""

import os
import csv
import time
from typing import Dict, Any, Optional


class TensorBoardLogger:

    def __init__(self, log_dir: str = "logs/tensorboard"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._writer = None
        self._csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None

        # Try TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
            print(f"  TensorBoard logging → {log_dir}")
            print(f"  Run: tensorboard --logdir logs/tensorboard")
        except ImportError:
            print(f"  TensorBoard not found — logging to CSV: {self._csv_path}")

        # Always init CSV fallback
        self._init_csv()

    def _init_csv(self):
        """Open CSV log file for writing."""
        write_header = not os.path.exists(self._csv_path)
        self._csv_file = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if write_header:
            self._csv_writer.writerow(["timestamp", "step", "key", "value"])
            self._csv_file.flush()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log a dict of {tag: scalar_value} at the given step.

        Example:
            logger.log_metrics({"train/loss": 0.42, "val/acc": 91.2}, step=50)
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for key, value in metrics.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue

            if self._writer is not None:
                self._writer.add_scalar(key, val, global_step=step)

            if self._csv_writer is not None:
                self._csv_writer.writerow([ts, step, key, f"{val:.6f}"])

        if self._csv_file:
            self._csv_file.flush()

    def log_lr(self, lr: float, step: int):
        """Convenience wrapper for logging learning rate."""
        self.log_metrics({"train/lr": lr}, step=step)

    def log_image(self, tag: str, image_tensor, step: int):
        """Log an image tensor (C, H, W) or (H, W). Silently skips if no TB."""
        if self._writer is not None:
            self._writer.add_image(tag, image_tensor, global_step=step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram of tensor values."""
        if self._writer is not None:
            self._writer.add_histogram(tag, values, global_step=step)

    def log_text(self, tag: str, text: str, step: int):
        """Log a text string."""
        if self._writer is not None:
            self._writer.add_text(tag, text, global_step=step)

    def close(self):
        """Flush and close the logger."""
        if self._writer is not None:
            self._writer.close()
        if self._csv_file is not None:
            self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ──────────────────────────────────────────────────────────────
    # Convenience: read CSV back for display
    # ──────────────────────────────────────────────────────────────

    def read_metrics(self, key_filter: Optional[str] = None) -> list:
        """
        Read all logged metrics from CSV.

        Args:
            key_filter: if given, only return rows where key contains this string.
        Returns:
            list of dicts {timestamp, step, key, value}
        """
        rows = []
        if not os.path.exists(self._csv_path):
            return rows

        with open(self._csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if key_filter and key_filter not in row.get("key", ""):
                    continue
                rows.append(row)
        return rows

    def summary(self) -> str:
        """Return a human-readable summary of the latest logged metrics."""
        if not os.path.exists(self._csv_path):
            return "No metrics logged yet."

        latest: Dict[str, str] = {}
        with open(self._csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                latest[row.get("key", "")] = row.get("value", "")

        lines = ["Latest metrics:"]
        for k, v in sorted(latest.items()):
            lines.append(f"  {k:<30} {v}")
        return "\n".join(lines)
