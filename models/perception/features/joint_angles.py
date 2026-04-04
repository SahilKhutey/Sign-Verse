"""
Joint angle calculation from pose landmarks.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

class JointAngleCalculator:
    """Calculates joint angles from pose landmarks."""
    
    # MediaPipe Pose landmark indices
    BODY_LANDMARK_INDICES = {
        'shoulder': [11, 12],      # Left and right shoulders
        'elbow': [13, 14],         # Left and right elbows
        'wrist': [15, 16],         # Left and right wrists
        'hip': [23, 24],           # Left and right hips
        'knee': [25, 26],          # Left and right knees
        'ankle': [27, 28],         # Left and right ankles
        'head': [0],               # Nose
        'neck': [1],               # Neck base
    }
    
    def __init__(self, angle_units: str = "degrees"):
        """
        Initialize joint angle calculator.
        
        Args:
            angle_units: Output angle units ('degrees' or 'radians')
        """
        self.angle_units = angle_units
    
    def calculate_angles(self, body_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate joint angles from body landmarks.
        
        Args:
            body_landmarks: Body landmarks array [33, 3]
            
        Returns:
            Dictionary of joint angles
        """
        if body_landmarks is None or len(body_landmarks) < 33:
            return {}
        
        angles = {}
        
        # Shoulder angles
        angles['left_shoulder'] = self._calculate_angle(
            body_landmarks[11], body_landmarks[13], body_landmarks[23]  # shoulder, elbow, hip
        )
        angles['right_shoulder'] = self._calculate_angle(
            body_landmarks[12], body_landmarks[14], body_landmarks[24]
        )
        
        # Elbow angles
        angles['left_elbow'] = self._calculate_angle(
            body_landmarks[13], body_landmarks[15], body_landmarks[11]  # elbow, wrist, shoulder
        )
        angles['right_elbow'] = self._calculate_angle(
            body_landmarks[14], body_landmarks[16], body_landmarks[12]
        )
        
        # Hip angles
        angles['left_hip'] = self._calculate_angle(
            body_landmarks[23], body_landmarks[25], body_landmarks[11]  # hip, knee, shoulder
        )
        angles['right_hip'] = self._calculate_angle(
            body_landmarks[24], body_landmarks[26], body_landmarks[12]
        )
        
        # Knee angles
        angles['left_knee'] = self._calculate_angle(
            body_landmarks[25], body_landmarks[27], body_landmarks[23]  # knee, ankle, hip
        )
        angles['right_knee'] = self._calculate_angle(
            body_landmarks[26], body_landmarks[28], body_landmarks[24]
        )
        
        # Neck angle (simplified)
        angles['neck'] = self._calculate_angle(
            body_landmarks[1], body_landmarks[0], body_landmarks[11]  # neck base, nose, shoulder
        )
        
        return angles
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate angle between vectors BA and BC.
        
        Args:
            a: Point A
            b: Point B (vertex)
            c: Point C
            
        Returns:
            Angle in specified units
        """
        ba = a - b
        bc = c - b
        
        # Normalize vectors
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        
        if ba_norm == 0 or bc_norm == 0:
            return 0.0
        
        ba_unit = ba / ba_norm
        bc_unit = bc / bc_norm
        
        # Calculate dot product and angle
        cosine_angle = np.dot(ba_unit, bc_unit)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        
        if self.angle_units == "degrees":
            return np.degrees(angle_rad)
        else:
            return angle_rad
    
    def calculate_hand_angles(self, hand_landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate hand joint angles."""
        # Similar implementation for hand landmarks
        # This would calculate finger joint angles
        return {}
    
    def calculate_relative_angles(self, 
                                current_angles: Dict[str, float],
                                reference_angles: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate angles relative to reference pose.
        
        Args:
            current_angles: Current joint angles
            reference_angles: Reference joint angles
            
        Returns:
            Relative angle differences
        """
        relative_angles = {}
        for joint, angle in current_angles.items():
            if joint in reference_angles:
                relative_angles[joint] = angle - reference_angles[joint]
            else:
                relative_angles[joint] = angle
        
        return relative_angles
