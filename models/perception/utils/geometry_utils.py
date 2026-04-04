"""
Geometry calculation utilities for pose landmarks using vectorized numpy operations.
"""
import numpy as np
from typing import List, Dict, Any, Optional

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points or arrays of points."""
    if p1 is None or p2 is None:
        return 0.0
    return np.linalg.norm(p1 - p2)

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the angle at p2 between p1-p2 and p2-p3 using vectorized operations.
    
    Args:
        p1, p2, p3: Points as numpy arrays [3]
        
    Returns:
        Angle in degrees [0, 180]
    """
    if any(p is None for p in [p1, p2, p3]):
        return 0.0
        
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    unit_v1 = v1 / v1_norm
    unit_v2 = v2 / v2_norm
    
    # Calculate dot product and angle
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    return float(np.degrees(angle))

def calculate_centroid(points: np.ndarray) -> np.ndarray:
    """Calculate the geometric center of a point cloud array [N, 3]."""
    if points is None or len(points) == 0:
        return np.zeros(3)
        
    return np.mean(points, axis=0)

def calculate_bounding_box(points: np.ndarray) -> np.ndarray:
    """Calculate the bounding box [x1, y1, x2, y2] of a point cloud array [N, 2/3]."""
    if points is None or len(points) == 0:
        return np.zeros(4)
        
    min_pt = np.min(points[:, :2], axis=0)
    max_pt = np.max(points[:, :2], axis=0)
    
    return np.array([min_pt[0], min_pt[1], max_pt[0], max_pt[1]])
