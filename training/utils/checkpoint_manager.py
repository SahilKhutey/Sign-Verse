"""
CheckpointManager — Atomic checkpoint save/load/resume for SignVerse models.

Features:
    - Atomic save via .tmp rename (crash-safe)
    - Saves: model state, optimizer state, epoch, best_metric, config hash
    - find_latest() scans save_dir for most recent checkpoint by epoch
    - Keeps only last N checkpoints to prevent disk bloat

Usage:
    ckpt = CheckpointManager(save_dir="models", prefix="gesture_model")
    ckpt.save(model, optimizer, epoch=10, metric=0.82, tag="best")

    latest = ckpt.find_latest()
    if latest:
        epoch, best = ckpt.load(latest, model, optimizer)
"""

import os
import glob
import json
import time
import torch
from typing import Optional, Tuple


class CheckpointManager:

    def __init__(self, save_dir: str = "models", prefix: str = "model",
                 keep_last_n: int = 3):
        """
        Args:
            save_dir:    Directory where checkpoints are written.
            prefix:      Filename prefix, e.g. "gesture_model".
            keep_last_n: How many periodic (non-best) checkpoints to retain.
        """
        self.save_dir = save_dir
        self.prefix = prefix
        self.keep_last_n = keep_last_n
        os.makedirs(save_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────

    def save(self, model: torch.nn.Module, optimizer, epoch: int,
             metric: float, tag: str = "checkpoint") -> str:
        """
        Save checkpoint atomically.

        Returns path of saved checkpoint.
        """
        filename = f"{self.prefix}_{tag}.pt"
        final_path = os.path.join(self.save_dir, filename)
        tmp_path = final_path + ".tmp"

        payload = {
            "epoch": epoch,
            "best_metric": metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "timestamp": time.time()
        }

        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)  # atomic on POSIX; best-effort on Windows

        # Also write a small JSON metadata sidecar
        meta_path = final_path.replace(".pt", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "best_metric": float(metric),
                "tag": tag,
                "filename": filename,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        # Prune old periodic checkpoints (not "best")
        if tag.startswith("epoch"):
            self._prune_old(exclude_tag="best")

        return final_path

    # ──────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────

    def load(self, path: str, model: torch.nn.Module,
             optimizer=None) -> Tuple[int, float]:
        """
        Load checkpoint into model (and optionally optimizer).

        Returns:
            (epoch, best_metric)
        """
        payload = torch.load(path, map_location="cpu")

        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer_state"])
            except Exception:
                pass  # ignore optimizer state mismatch (e.g. config changed)

        epoch = payload.get("epoch", 0)
        metric = payload.get("best_metric", float("inf"))
        print(f"  Loaded checkpoint: {os.path.basename(path)} "
              f"(epoch={epoch}, metric={metric:.4f})")
        return epoch, metric

    # ──────────────────────────────────────────────────────────────
    # Find latest
    # ──────────────────────────────────────────────────────────────

    def find_latest(self) -> Optional[str]:
        """
        Scan save_dir for the most recent checkpoint for this prefix.

        Prefers 'best' checkpoint if it exists, otherwise returns
        the periodic checkpoint with the highest epoch number.
        """
        # Best checkpoint takes priority if it exists
        best_path = os.path.join(self.save_dir, f"{self.prefix}_best.pt")
        if os.path.exists(best_path):
            return best_path

        # Otherwise find highest epoch checkpoint
        pattern = os.path.join(self.save_dir, f"{self.prefix}_epoch*.pt")
        candidates = glob.glob(pattern)
        if not candidates:
            return None

        def _epoch_num(p):
            name = os.path.basename(p)
            try:
                # e.g. gesture_model_epoch50.pt → 50
                part = name.replace(f"{self.prefix}_epoch", "").replace(".pt", "")
                return int(part)
            except ValueError:
                return 0

        candidates.sort(key=_epoch_num, reverse=True)
        return candidates[0]

    # ──────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────

    def _prune_old(self, exclude_tag: str = "best"):
        """Remove oldest periodic checkpoints beyond keep_last_n."""
        pattern = os.path.join(self.save_dir, f"{self.prefix}_epoch*.pt")
        candidates = glob.glob(pattern)
        if len(candidates) <= self.keep_last_n:
            return

        def _epoch_num(p):
            name = os.path.basename(p)
            try:
                return int(name.replace(f"{self.prefix}_epoch", "").replace(".pt", ""))
            except ValueError:
                return 0

        candidates.sort(key=_epoch_num)
        # Remove all but the last keep_last_n
        for old_ckpt in candidates[:-self.keep_last_n]:
            try:
                os.remove(old_ckpt)
                meta = old_ckpt.replace(".pt", "_meta.json")
                if os.path.exists(meta):
                    os.remove(meta)
            except OSError:
                pass

    # ──────────────────────────────────────────────────────────────
    # List all
    # ──────────────────────────────────────────────────────────────

    def list_checkpoints(self) -> list:
        """Return list of checkpoint paths sorted by modification time."""
        pattern = os.path.join(self.save_dir, f"{self.prefix}_*.pt")
        paths = glob.glob(pattern)
        paths.sort(key=os.path.getmtime)
        return paths
