"""
Hand gesture detection heuristics using vectorized numpy operations.
"""
import numpy as np
from typing import Dict, List, Any, Optional

def detect_hand_gestures(left_hand: Optional[np.ndarray], 
                        right_hand: Optional[np.ndarray]) -> List[str]:
    """
    Detect hand gestures for both hands.
    
    Args:
        left_hand: Landmarks for left hand [21, 3]
        right_hand: Landmarks for right hand [21, 3]
        
    Returns:
        List of detected gesture labels
    """
    gestures = []
    
    if left_hand is not None:
        if is_open_palm(left_hand):
            gestures.append('left_hand_open')
        if is_fist(left_hand):
            gestures.append('left_hand_fist')
            
    if right_hand is not None:
        if is_open_palm(right_hand):
            gestures.append('right_hand_open')
        if is_fist(right_hand):
            gestures.append('right_hand_fist')
            
    return gestures

def is_open_palm(landmarks: np.ndarray) -> bool:
    """Check if hand is in an open palm configuration using vectorized distance."""
    if landmarks is None or len(landmarks) < 21:
        return False
    
    finger_tips = np.array([8, 12, 16, 20])
    finger_mcps = np.array([5, 9, 13, 17])
    wrist = landmarks[0]
    
    # Vectorized Euclidean distance from wrist
    dist_tips = np.linalg.norm(landmarks[finger_tips] - wrist, axis=1)
    dist_mcps = np.linalg.norm(landmarks[finger_mcps] - wrist, axis=1)
    
    # Count fingers where tip is significantly further than MCP from wrist
    extended = np.sum(dist_tips > dist_mcps * 1.1)
    return extended >= 3

def is_fist(landmarks: np.ndarray) -> bool:
    """Check if hand is in a fist configuration using vectorized distance."""
    if landmarks is None or len(landmarks) < 21:
        return False
        
    finger_tips = np.array([8, 12, 16, 20])
    finger_mcps = np.array([5, 9, 13, 17])
    wrist = landmarks[0]
    
    # If tip is closer to wrist than MCP, it's curled
    dist_tips = np.linalg.norm(landmarks[finger_tips] - wrist, axis=1)
    dist_mcps = np.linalg.norm(landmarks[finger_mcps] - wrist, axis=1)
    
    curled = np.sum(dist_tips < dist_mcps)
    return curled >= 3
