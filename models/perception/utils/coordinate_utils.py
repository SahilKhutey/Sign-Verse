"""
Coordinate transformation and normalization utilities.
"""
import numpy as np
from typing import Tuple, Optional

def normalize_coordinates(points: np.ndarray, 
                         image_shape: Tuple[int, int],
                         normalize_z: bool = False) -> np.ndarray:
    """
    Normalize coordinates from pixel space to normalized [0, 1] range.
    
    Args:
        points: Input points array [N, 2] or [N, 3]
        image_shape: (height, width) of the image
        normalize_z: Whether to normalize Z coordinates
        
    Returns:
        Normalized points array
    """
    if points is None or len(points) == 0:
        return points
    
    height, width = image_shape
    normalized = points.copy().astype(float)
    
    # Normalize X and Y
    if len(normalized.shape) == 2:
        normalized[:, 0] /= width   # X coordinate
        normalized[:, 1] /= height  # Y coordinate
        
        # Normalize Z if requested and available
        if normalize_z and normalized.shape[1] >= 3:
            # Use average of width and height for Z normalization
            avg_dim = (width + height) / 2
            normalized[:, 2] /= avg_dim
    
    return normalized

def pixel_to_world(points: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist_coeffs: Optional[np.ndarray] = None,
                  z_const: float = 1.0) -> np.ndarray:
    """
    Convert pixel coordinates to world coordinates (simplified).
    
    Args:
        points: Pixel coordinates [N, 2]
        camera_matrix: Camera intrinsic matrix [3, 3]
        dist_coeffs: Distortion coefficients
        z_const: Assumed Z coordinate for conversion
        
    Returns:
        World coordinates [N, 3]
    """
    if points is None or len(points) == 0:
        return points
    
    # Simple conversion assuming fixed Z
    # In production, you'd use proper camera calibration and triangulation
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    
    # Convert to homogeneous coordinates
    homogeneous = np.ones((len(points), 3))
    homogeneous[:, :2] = points
    
    # Transform to camera coordinates
    camera_coords = homogeneous @ inv_camera_matrix.T
    
    # Scale by Z constant
    world_coords = camera_coords * z_const
    
    return world_coords

def world_to_pixel(points: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert world coordinates to pixel coordinates.
    
    Args:
        points: World coordinates [N, 3]
        camera_matrix: Camera intrinsic matrix [3, 3]
        dist_coeffs: Distortion coefficients
        
    Returns:
        Pixel coordinates [N, 2]
    """
    if points is None or len(points) == 0:
        return points
    
    # Project 3D points to 2D
    homogeneous = points @ camera_matrix.T
    pixel_coords = homogeneous[:, :2] / (homogeneous[:, 2:3] + 1e-6)
    
    # Apply distortion correction if provided
    if dist_coeffs is not None:
        # Simplified distortion correction
        # In production, use OpenCV's projectPoints
        pass
    
    return pixel_coords

def calculate_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Calculate rotation matrix around given axis.
    
    Args:
        axis: Rotation axis vector [3]
        angle: Rotation angle in radians
        
    Returns:
        Rotation matrix [3, 3]
    """
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1 - cos_a
    
    # Rodrigues' rotation formula
    rotation_matrix = np.array([
        [cos_a + axis[0]**2 * one_minus_cos,
         axis[0]*axis[1]*one_minus_cos - axis[2]*sin_a,
         axis[0]*axis[2]*one_minus_cos + axis[1]*sin_a],
        
        [axis[1]*axis[0]*one_minus_cos + axis[2]*sin_a,
         cos_a + axis[1]**2 * one_minus_cos,
         axis[1]*axis[2]*one_minus_cos - axis[0]*sin_a],
        
        [axis[2]*axis[0]*one_minus_cos - axis[1]*sin_a,
         axis[2]*axis[1]*one_minus_cos + axis[0]*sin_a,
         cos_a + axis[2]**2 * one_minus_cos]
    ])
    
    return rotation_matrix

def transform_points(points: np.ndarray, 
                    rotation: np.ndarray,
                    translation: np.ndarray) -> np.ndarray:
    """
    Apply rotation and translation to points.
    
    Args:
        points: Input points [N, 3]
        rotation: Rotation matrix [3, 3]
        translation: Translation vector [3]
        
    Returns:
        Transformed points [N, 3]
    """
    if points is None or len(points) == 0:
        return points
    
    # Apply rotation
    rotated = points @ rotation.T
    
    # Apply translation
    transformed = rotated + translation
    
    return transformed
