"""
Math and statistical utilities for perception processing.
"""
import numpy as np
from typing import List, Dict, Any, Optional

def calculate_rolling_average(values: np.ndarray, window_size: int = 5) -> float:
    """Calculate the moving average of a signal."""
    if len(values) < window_size:
        return np.mean(values)
    return np.mean(values[-window_size:])

def calculate_signal_variance(values: np.ndarray, window_size: int = 5) -> float:
    """Calculate the variance of a signal in a window."""
    if len(values) < window_size:
        return np.var(values)
    return np.var(values[-window_size:])

def exponential_moving_average(current: float, previous: float, alpha: float = 0.5) -> float:
    """Apply exponential moving average filter."""
    return alpha * current + (1 - alpha) * previous

def sigmoid(x: float) -> float:
    """Standard sigmoid function."""
    return 1 / (1 + np.exp(-x))
