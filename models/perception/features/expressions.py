"""
Facial expression detection heuristics using vectorized numpy operations.
"""
import numpy as np
from typing import Dict, List, Any, Optional

def classify_facial_expressions(face_landmarks: Optional[np.ndarray]) -> List[str]:
    """
    Classify facial expressions from face landmarks using vectorized aspect ratios.
    
    Args:
        face_landmarks: Standardized face landmarks [N, 3]
        
    Returns:
        List of detected expression labels
    """
    expressions = []
    
    if face_landmarks is None or len(face_landmarks) < 468:
        return expressions
        
    # Smile detection (Mouth Aspect Ratio)
    if is_smiling(face_landmarks):
        expressions.append('smile')
    
    # Eye openness (Eye Aspect Ratio)
    left_eye_open = is_eye_open(face_landmarks, 'left')
    right_eye_open = is_eye_open(face_landmarks, 'right')
    
    if not left_eye_open and not right_eye_open:
        expressions.append('eyes_closed')
    elif not left_eye_open:
        expressions.append('left_eye_closed')
    elif not right_eye_open:
        expressions.append('right_eye_closed')
        
    return expressions

def is_smiling(face_landmarks: np.ndarray) -> bool:
    """Detect smile using vectorized Mouth Aspect Ratio (MAR)."""
    # corners: 61, 291. upper: 13, lower: 14.
    p_l = face_landmarks[61]
    p_r = face_landmarks[291]
    p_u = face_landmarks[13]
    p_b = face_landmarks[14]
    
    mouth_width = np.linalg.norm(p_l - p_r)
    mouth_height = np.linalg.norm(p_u - p_b)
    
    ratio = mouth_width / (mouth_height + 1e-6)
    return ratio > 3.0

def is_eye_open(face_landmarks: np.ndarray, side: str) -> bool:
    """Detect if eye is open using vectorized Eye Aspect Ratio (EAR)."""
    if side == 'left':
        # Left Eye indices: 33 (L), 133 (R), 159 (Top), 145 (Bot)
        idx = np.array([33, 133, 159, 145])
    else:
        # Right Eye indices: 362 (L), 263 (R), 386 (Top), 374 (Bot)
        idx = np.array([362, 263, 386, 374])
        
    p1, p2, p3, p4 = face_landmarks[idx]
        
    width = np.linalg.norm(p1 - p2)
    height = np.linalg.norm(p3 - p4)
    
    ear = height / (width + 1e-6)
    return ear > 0.2
