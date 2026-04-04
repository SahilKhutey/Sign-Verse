"""
Normalization system for converting raw coordinates to ML-ready format.
Handles reference centering, proportional scaling, and coordinate alignment.
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
from scipy.spatial.transform import Rotation

from ..data_models.skeleton import SkeletonFrame, BodyJoints, Vector3D

@dataclass
class NormalizationParams:
    """Parameters for normalization."""
    scale_method: str = "height"  # "height", "torso", "fixed"
    reference_joint: str = "hip_center"
    forward_axis: str = "z"  # "x", "y", "z"
    up_axis: str = "y"       # "x", "y", "z"
    scale_to: float = 1.0    # Target scale
    
    # Coordinate system conventions
    coordinate_system: str = "right_handed"  # "right_handed", "left_handed"
    axis_directions: Dict[str, str] = None   # Custom axis directions

class PoseNormalizer:
    """Normalizes pose data for consistency and ML readiness."""
    
    def __init__(self, params: Optional[NormalizationParams] = None):
        """
        Initialize pose normalizer.
        
        Args:
            params: Normalization parameters
        """
        self.params = params or NormalizationParams()
        self.reference_points = {}
        
        # Default axis directions for right-handed system
        if self.params.axis_directions is None:
            self.params.axis_directions = {
                "x": "right",
                "y": "up", 
                "z": "forward"
            }
    
    def normalize_frame(self, frame: SkeletonFrame) -> SkeletonFrame:
        """
        Normalize a complete skeleton frame.
        
        Args:
            frame: Input SkeletonFrame
            
        Returns:
            Normalized SkeletonFrame
        """
        if not frame.body:
            return frame
        
        try:
            # Create a copy to avoid modifying original
            normalized_frame = frame.copy()
            
            # 1. Calculate reference point (usually hip center)
            reference_point = self._calculate_reference_point(frame.body)
            
            # 2. Calculate scale factor
            scale_factor = self._calculate_scale_factor(frame.body, reference_point)
            
            # 3. Calculate rotation to align with coordinate system
            rotation_matrix = self._calculate_alignment_rotation(frame.body, reference_point)
            
            # 4. Apply transformation to all joints
            normalized_body = self._transform_body(
                frame.body, reference_point, scale_factor, rotation_matrix
            )
            
            normalized_frame.body = normalized_body
            
            # 5. Normalize hands relative to body
            if frame.left_hand:
                normalized_frame.left_hand = self._transform_hand(
                    frame.left_hand, reference_point, scale_factor, rotation_matrix
                )
            
            if frame.right_hand:
                normalized_frame.right_hand = self._transform_hand(
                    frame.right_hand, reference_point, scale_factor, rotation_matrix
                )
            
            # 6. Normalize face relative to body
            if frame.face:
                normalized_frame.face = self._transform_face(
                    frame.face, reference_point, scale_factor, rotation_matrix
                )
            
            # Update confidence to reflect normalization quality
            normalized_frame.overall_confidence *= self._calculate_normalization_confidence(
                frame.body, normalized_body
            )
            
            return normalized_frame
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return frame
    
    def _calculate_reference_point(self, body: BodyJoints) -> np.ndarray:
        """Calculate reference point for normalization."""
        joints_dict = body.get_joints_dict()
        
        if self.params.reference_joint == "hip_center":
            # Use midpoint between left and right hips
            left_hip = joints_dict.get("left_hip")
            right_hip = joints_dict.get("right_hip")
            
            if left_hip and right_hip:
                left_pos = np.array([left_hip.x, left_hip.y, left_hip.z])
                right_pos = np.array([right_hip.x, right_hip.y, right_hip.z])
                return (left_pos + right_pos) / 2.0
            
            # Fallback to first available joint
            for joint in joints_dict.values():
                return np.array([joint.x, joint.y, joint.z])
        
        elif self.params.reference_joint in joints_dict:
            # Use specific joint
            joint = joints_dict[self.params.reference_joint]
            return np.array([joint.x, joint.y, joint.z])
        
        # Default to origin
        return np.zeros(3)
    
    def _calculate_scale_factor(self, body: BodyJoints, reference_point: np.ndarray) -> float:
        """Calculate scale factor based on body proportions."""
        joints_dict = body.get_joints_dict()
        
        if self.params.scale_method == "height":
            # Scale based on body height (shoulder to foot distance)
            shoulder = joints_dict.get("left_shoulder") or joints_dict.get("right_shoulder")
            foot = joints_dict.get("left_ankle") or joints_dict.get("right_ankle")
            
            if shoulder and foot:
                shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])
                foot_pos = np.array([foot.x, foot.y, foot.z])
                height = np.linalg.norm(shoulder_pos - foot_pos)
                return self.params.scale_to / height if height > 0 else 1.0
        
        elif self.params.scale_method == "torso":
            # Scale based on torso length (shoulder to hip distance)
            shoulder = joints_dict.get("left_shoulder") or joints_dict.get("right_shoulder")
            hip = joints_dict.get("left_hip") or joints_dict.get("right_hip")
            
            if shoulder and hip:
                shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])
                hip_pos = np.array([hip.x, hip.y, hip.z])
                torso_length = np.linalg.norm(shoulder_pos - hip_pos)
                return self.params.scale_to / torso_length if torso_length > 0 else 1.0
        
        # Fixed scale or fallback
        return self.params.scale_to
    
    def _calculate_alignment_rotation(self, body: BodyJoints, reference_point: np.ndarray) -> np.ndarray:
        """Calculate rotation to align body with coordinate system."""
        joints_dict = body.get_joints_dict()
        
        # Calculate forward direction (from hip to shoulders)
        left_shoulder = joints_dict.get("left_shoulder")
        right_shoulder = joints_dict.get("right_shoulder")
        left_hip = joints_dict.get("left_hip")
        right_hip = joints_dict.get("right_hip")
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            left_pos = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
            right_pos = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
            l_hip_pos = np.array([left_hip.x, left_hip.y, left_hip.z])
            r_hip_pos = np.array([right_hip.x, right_hip.y, right_hip.z])
            
            # Shoulder vector (right to left)
            shoulder_vec = left_pos - right_pos
            
            # Up direction (approximate from hip to shoulders)
            hip_mid = (l_hip_pos + r_hip_pos) / 2.0
            shoulder_mid = (left_pos + right_pos) / 2.0
            spine_vec = shoulder_mid - hip_mid
            
            # Calculate forward direction (cross product of shoulder and spine)
            forward_vec = np.cross(shoulder_vec, spine_vec)
            
            # Normalize vectors
            shoulder_vec_norm = shoulder_vec / np.linalg.norm(shoulder_vec)
            spine_vec_norm = spine_vec / np.linalg.norm(spine_vec)
            forward_vec_norm = forward_vec / np.linalg.norm(forward_vec)
            
            # Recalculate up vector for orthogonality
            up_vec_norm = np.cross(forward_vec_norm, shoulder_vec_norm)
            
            # Create rotation matrix (X=Right, Y=Up, Z=Forward)
            rotation_matrix = np.column_stack([
                shoulder_vec_norm,  # X
                up_vec_norm,        # Y
                forward_vec_norm     # Z
            ])
            
            return rotation_matrix
        
        # Identity matrix if alignment fails
        return np.eye(3)
    
    def _transform_body(self, body: BodyJoints, reference_point: np.ndarray,
                       scale_factor: float, rotation_matrix: np.ndarray) -> BodyJoints:
        """Transform body joints to normalized coordinate system."""
        normalized_body = BodyJoints()
        joints_dict = body.get_joints_dict()
        
        for joint_name, joint in joints_dict.items():
            original_pos = np.array([joint.x, joint.y, joint.z])
            
            normalized_pos = self._apply_transformation(
                original_pos, reference_point, scale_factor, rotation_matrix
            )
            
            normalized_joint = Vector3D(
                x=float(normalized_pos[0]),
                y=float(normalized_pos[1]),
                z=float(normalized_pos[2]),
                confidence=joint.confidence
            )
            
            setattr(normalized_body, joint_name, normalized_joint)
        
        return normalized_body
    
    def _transform_hand(self, hand, reference_point: np.ndarray,
                       scale_factor: float, rotation_matrix: np.ndarray):
        """Transform hand joints relative to body."""
        from ..data_models.skeleton import HandJoints
        normalized_hand = HandJoints()
        
        # Iterate over all possible hand joint names
        for field_name in hand.__dict__.keys():
            joint = getattr(hand, field_name)
            if joint is not None and isinstance(joint, Vector3D):
                original_pos = np.array([joint.x, joint.y, joint.z])
                normalized_pos = self._apply_transformation(
                    original_pos, reference_point, scale_factor, rotation_matrix
                )
                
                setattr(normalized_hand, field_name, Vector3D(
                    x=float(normalized_pos[0]),
                    y=float(normalized_pos[1]),
                    z=float(normalized_pos[2]),
                    confidence=joint.confidence
                ))
        
        return normalized_hand
    
    def _transform_face(self, face, reference_point: np.ndarray,
                       scale_factor: float, rotation_matrix: np.ndarray):
        """Transform face landmarks relative to body."""
        from ..data_models.skeleton import FaceLandmarks
        normalized_face = FaceLandmarks()
        normalized_landmarks = []
        
        for landmark in face.landmarks:
            original_pos = np.array([landmark.x, landmark.y, landmark.z])
            normalized_pos = self._apply_transformation(
                original_pos, reference_point, scale_factor, rotation_matrix
            )
            
            normalized_landmarks.append(Vector3D(
                x=float(normalized_pos[0]),
                y=float(normalized_pos[1]),
                z=float(normalized_pos[2]),
                confidence=landmark.confidence
            ))
        
        normalized_face.landmarks = normalized_landmarks
        normalized_face.expression = face.expression
        
        return normalized_face
    
    def _apply_transformation(self, position: np.ndarray, reference_point: np.ndarray,
                            scale_factor: float, rotation_matrix: np.ndarray) -> np.ndarray:
        """Apply complete transformation to a point."""
        # 1. Center relative to reference point
        centered = position - reference_point
        
        # 2. Scale
        scaled = centered * scale_factor
        
        # 3. Rotate to align with coordinate system
        # We need the inverse (transpose) rotation to align original coords TO canonical basis
        rotated = rotation_matrix.T @ scaled
        
        return rotated
    
    def _calculate_normalization_confidence(self, original_body: BodyJoints, 
                                          normalized_body: BodyJoints) -> float:
        """Calculate confidence in normalization quality."""
        joints_dict = normalized_body.get_joints_dict()
        if not joints_dict:
            return 0.5
        
        # Check shoulder width (should be ~0.4-0.6 in normalized units)
        l_shoulder = joints_dict.get("left_shoulder")
        r_shoulder = joints_dict.get("right_shoulder")
        
        if l_shoulder and r_shoulder:
            width = np.linalg.norm(l_shoulder.to_array() - r_shoulder.to_array())
            if width < 0.2 or width > 1.0:
                return 0.4
        
        return 0.9

# Pre-configured normalization presets
NORMALIZATION_PRESETS = {
    "ml_training": NormalizationParams(
        scale_method="height",
        reference_joint="hip_center",
        scale_to=1.0,
        coordinate_system="right_handed"
    ),
    "robotics": NormalizationParams(
        scale_method="torso", 
        reference_joint="hip_center",
        scale_to=0.5,
        coordinate_system="right_handed"
    )
}

def get_normalizer(preset_name: str = "ml_training") -> PoseNormalizer:
    """Get a pre-configured normalizer."""
    if preset_name not in NORMALIZATION_PRESETS:
        logger.warning(f"Unknown preset {preset_name}, using ml_training")
        preset_name = "ml_training"
    return PoseNormalizer(NORMALIZATION_PRESETS[preset_name])
