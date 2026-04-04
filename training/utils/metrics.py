"""
Metrics Utilities

These metrics are used in trainers for quick sanity-checks and validation
without pulling in external libraries.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
    ignore_index: Optional[int] = None,
) -> float:
    """
    Compute top-k accuracy for classification logits.

    Args:
        logits: (batch, num_classes)
        targets: (batch,)
    """
    if logits.numel() == 0:
        return 0.0
    if ignore_index is not None:
        mask = targets != ignore_index
        logits = logits[mask]
        targets = targets[mask]
        if targets.numel() == 0:
            return 0.0

    topk = logits.topk(k=min(int(k), logits.size(-1)), dim=-1).indices  # (batch, k)
    correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return float(correct) * 100.0


def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
) -> float:
    """
    Token-level accuracy for sequence logits.

    Args:
        logits: (batch, seq, vocab)
        targets: (batch, seq)
    """
    preds = logits.argmax(dim=-1)
    mask = targets != int(pad_id)
    if mask.sum().item() == 0:
        return 0.0
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return float(correct) / float(total) * 100.0


def perplexity(loss: torch.Tensor, clamp_max: float = 10000.0) -> float:
    try:
        ppl = torch.exp(loss.detach()).item()
        return float(min(ppl, clamp_max))
    except Exception:
        return float("inf")
