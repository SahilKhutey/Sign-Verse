"""
Performance Monitor

Tiny helpers to measure latency of pipeline stages.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


@contextmanager
def timed(metrics: Dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = (time.perf_counter() - start) * 1000.0


class PerfCounter:
    def __init__(self):
        self.metrics: Dict[str, float] = {}

    def reset(self) -> None:
        self.metrics.clear()

    def time(self, key: str):
        return timed(self.metrics, key)

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        return self.metrics.get(key, default)
