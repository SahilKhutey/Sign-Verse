"""
Acceleration calculation for pose landmarks using vectorized numpy operations.
"""
import numpy as np
from typing import Dict, List, Any, Optional

def calculate_joint_accelerations(current_velocities: Dict[str, np.ndarray],
                                 previous_velocities: Dict[str, np.ndarray],
                                 dt: float) -> Dict[str, np.ndarray]:
    """
    Calculate accelerations for each landmark type between two frames.
    
    Args:
        current_velocities: Dictionary mapping landmark type to current velocity [N] array
        previous_velocities: Dictionary mapping landmark type to previous velocity [N] array
        dt: Time difference between frames
        
    Returns:
        Dictionary mapping landmark type to numpy array of accelerations [N]
    """
    accelerations = {}
    
    if dt <= 0:
        return accelerations
        
    for landmark_type, velocities in current_velocities.items():
        if landmark_type not in previous_velocities:
            continue
            
        prev_velocities = previous_velocities[landmark_type]
        if prev_velocities is None or velocities.shape != prev_velocities.shape:
            continue
            
        # (v2 - v1) / dt
        type_accelerations = (velocities - prev_velocities) / dt
        accelerations[landmark_type] = type_accelerations
        
    return accelerations
