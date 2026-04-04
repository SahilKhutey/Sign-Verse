"""
Feature extraction models for pose landmarks.
"""
from .velocity import calculate_joint_velocities
from .acceleration import calculate_joint_accelerations
from .gestures import detect_hand_gestures
from .expressions import classify_facial_expressions
from .joint_angles import JointAngleCalculator
