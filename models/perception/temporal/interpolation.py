"""
Interpolation and missing data handling for pose landmark sequences.
"""
import numpy as np
from typing import Dict, List, Any, Optional

def interpolate_missing_landmarks(landmarks: np.ndarray, 
                                 confidences: np.ndarray, 
                                 threshold: float = 0.3) -> np.ndarray:
    """
    Handle missing or low-confidence data in a single landmark array.
    
    Args:
        landmarks: Landmark data array [N, 3]
        confidences: Confidence scores array [N]
        threshold: Confidence threshold for validity
        
    Returns:
        Landmark array with missing data filled (nearest-neighbor)
    """
    if landmarks is None or len(landmarks) == 0:
        return landmarks
        
    low_confidence = confidences < threshold
    if not np.any(low_confidence):
        return landmarks
        
    # Interpolate missing points (simple nearest neighbor)
    valid_indices = np.where(confidences >= threshold)[0]
    if len(valid_indices) == 0:
        return landmarks
        
    interpolated = landmarks.copy()
    for i in range(len(confidences)):
        if confidences[i] < threshold:
            # Find nearest valid index
            nearest = valid_indices[np.argmin(np.abs(valid_indices - i))]
            interpolated[i] = landmarks[nearest]
            
    return interpolated

def interpolate_full_pose(pose_landmarks: Dict[str, np.ndarray],
                         pose_confidences: Dict[str, np.ndarray],
                         threshold: float = 0.3) -> Dict[str, np.ndarray]:
    """
    Handle missing or low-confidence data across all landmark types.
    
    Args:
        pose_landmarks: Dictionary mapping type to landmark arrays
        pose_confidences: Dictionary mapping type to confidence arrays
        threshold: Confidence threshold for validity
        
    Returns:
        Dictionary with interpolated landmark arrays
    """
    interpolated_pose = {}
    for landmark_type, points in pose_landmarks.items():
        if landmark_type in pose_confidences:
            interpolated_pose[landmark_type] = interpolate_missing_landmarks(
                points, pose_confidences[landmark_type], threshold
            )
        else:
            interpolated_pose[landmark_type] = points
            
    return interpolated_pose
