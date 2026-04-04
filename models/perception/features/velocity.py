"""
Velocity calculation for pose landmarks using vectorized numpy operations.
"""
import numpy as np
from typing import Dict, List, Any, Optional

def calculate_joint_velocities(current_pose: Dict[str, np.ndarray], 
                             previous_pose: Dict[str, np.ndarray], 
                             dt: float) -> Dict[str, np.ndarray]:
    """
    Calculate velocities for each landmark type between two frames.
    
    Args:
        current_pose: Current pose data (landmark type -> [N, 3] array)
        previous_pose: Previous pose data (landmark type -> [N, 3] array)
        dt: Time difference between frames
        
    Returns:
        Dictionary mapping landmark type to numpy array of velocities [N]
    """
    velocities = {}
    
    if dt <= 0:
        return velocities
        
    for landmark_type, landmarks in current_pose.items():
        if landmarks is None or landmark_type not in previous_pose:
            continue
            
        prev_landmarks = previous_pose[landmark_type]
        if prev_landmarks is None or landmarks.shape != prev_landmarks.shape:
            continue
            
        # Vectorized Euclidean distance / dt
        # norm over Axis 1 (the 3 coordinates)
        type_velocities = np.linalg.norm((landmarks - prev_landmarks), axis=1) / dt
        velocities[landmark_type] = type_velocities
            
    return velocities
