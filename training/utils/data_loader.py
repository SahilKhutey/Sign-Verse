"""
DataLoader Helpers

Central place for small DataLoader conventions used across trainers.
"""

from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )
