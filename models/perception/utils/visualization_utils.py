"""
Visualization and drawing utilities for pose landmarks and detections.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

def draw_bbox(frame: np.ndarray, bbox: np.ndarray, label: str = "Person", color: tuple = (0, 255, 0)):
    """Draw bounding box on frame."""
    if bbox is None or len(bbox) < 4:
        return frame
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def draw_landmarks(frame: np.ndarray, 
                   landmarks: np.ndarray, 
                   color: tuple = (255, 0, 0), 
                   radius: int = 3,
                   normalized: bool = True):
    """
    Draw landmarks as points on frame using vectorized numpy slicing.
    
    Args:
        frame: BGR image
        landmarks: [N, 2] or [N, 3] array
        color: BGR color tuple
        radius: Point radius
        normalized: Whether landmarks are in [0, 1] (True) or pixel (False) space
    """
    if landmarks is None or len(landmarks) == 0:
        return frame
        
    h, w = frame.shape[:2]
    
    # Vectorized scale-and-cast
    if normalized:
        pts = (landmarks[:, :2] * [w, h]).astype(int)
    else:
        pts = landmarks[:, :2].astype(int)
        
    for p in pts:
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(frame, (p[0], p[1]), radius, color, -1)
            
    return frame

def draw_skeleton(frame: np.ndarray, 
                  landmarks: np.ndarray, 
                  connections: List[tuple], 
                  color: tuple = (0, 255, 255),
                  thickness: int = 2,
                  normalized: bool = True):
    """
    Draw skeleton lines connecting landmarks using vectorized numpy scaling.
    
    Args:
        frame: BGR image
        landmarks: [N, 2] or [N, 3] array
        connections: List of (start_idx, end_idx) tuples
        color: BGR color tuple
        thickness: Line thickness
        normalized: Whether landmarks are in [0, 1] or pixel space
    """
    if landmarks is None or len(landmarks) == 0:
        return frame
        
    h, w = frame.shape[:2]
    
    # Vectorized scale-and-cast
    if normalized:
        pts = (landmarks[:, :2] * [w, h]).astype(int)
    else:
        pts = landmarks[:, :2].astype(int)
        
    for start_idx, end_idx in connections:
        if start_idx < len(pts) and end_idx < len(pts):
            p1 = pts[start_idx]
            p2 = pts[end_idx]
            
            # Simple bounds check for lines
            if all(0 <= c < w if i%2==0 else 0 <= c < h for i, c in enumerate([p1[0], p1[1], p2[0], p2[1]])):
                cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), color, thickness)
                
    return frame
