"""
SMPL & SMPL-X Parametric Human Body Model
Standardized parametric representation for sign language motion capture.
Normalization: Pose (relative rotation) vs. Shape (identity).
"""

import numpy as np
import logging

class SMPLModel:
    """
    Standard SMPL (Body only) parametric model.
    Joints: 24
    Pose params: 72 (24 joints * 3 axis-angles)
    Shape params: 10 (beta)
    """
    def __init__(self, model_type="smpl"):
        self.model_type = model_type
        self.num_joints = 24
        self.pose_dim = 72
        self.shape_dim = 10
        
    def get_default_pose(self):
        """Returns zero pose (T-pose)."""
        return np.zeros(self.pose_dim, dtype=np.float32)
        
    def get_default_shape(self):
        """Returns zero shape (mean body)."""
        return np.zeros(self.shape_dim, dtype=np.float32)

    def project_to_mean_body(self, pose, shape=None):
        """
        Normalizes a pose by projecting it onto the mean body shape (beta=0).
        This decouples human size from the 'sign geometry.'
        """
        # For pure geometric normalization, we simply return the pose 
        # while explicitly discarding the shape variance.
        return {
            "pose": pose,
            "shape": np.zeros(self.shape_dim, dtype=np.float32)
        }


class SMPLXModel(SMPLModel):
    """
    Extended SMPL-X (Body + Hands + Face) parametric model.
    Joints: 54
    Pose params: 162 (54 joints * 3 axis-angles)
    Shape params: 10 (beta) + 10 (expression)
    """
    def __init__(self):
        super().__init__(model_type="smplx")
        self.num_joints = 54
        self.pose_dim = 162  # 54 * 3
        self.shape_dim = 10
        self.exp_dim = 10
        
    def get_default_pose(self):
        return np.zeros(self.pose_dim, dtype=np.float32)
        
    def get_default_shape(self):
        return np.zeros(self.shape_dim, dtype=np.float32)

    def get_default_expression(self):
        return np.zeros(self.exp_dim, dtype=np.float32)

    def extract_expressive_features(self, pose_params):
        """
        Splits SMPL-X pose parameters into Body, Hand, and Jaw segments.
        Useful for modular robotics mapping (body to torso, hands to grippers).
        """
        # Joint mapping index ranges (Standard SMPL-X indexes)
        # 0-21: Body joints (some overlaps)
        # 22-36: Left hand joints (15 joints)
        # 37-51: Right hand joints (15 joints)
        # 52: Jaw
        # 53, 54: Eyes
        
        segments = {
            "body": pose_params[0:22*3],
            "left_hand": pose_params[22*3:37*3],
            "right_hand": pose_params[37*3:52*3],
            "jaw": pose_params[52*3:53*3],
            "eyes": pose_params[53*3:]
        }
        return segments

    def project_to_mean_body(self, pose, shape=None, expression=None):
        """
        Normalizes SMPL-X expressive parameters onto a standard body/face.
        """
        return {
            "pose": pose,
            "shape": np.zeros(self.shape_dim, dtype=np.float32),
            "expression": np.zeros(self.exp_dim, dtype=np.float32)
        }
