"""
Unified skeleton data structure for consistent pose representation.
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime
from loguru import logger

class LandmarkType(str, Enum):
    """Standardized landmark types."""
    BODY = "body"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    FACE = "face"
    HEAD_POSE = "head_pose"

class ExpressionType(str, Enum):
    """Facial expression types."""
    SMILE = "smile"
    FROWN = "frown"
    SURPRISE = "surprise"
    BLINK_LEFT = "blink_left"
    BLINK_RIGHT = "blink_right"
    MOUTH_OPEN = "mouth_open"
    EYEBROWS_RAISED = "eyebrows_raised"

class JointType(str, Enum):
    """Standardized body joint names."""
    # Body joints (MediaPipe Pose 33 points)
    NOSE = "nose"
    LEFT_EYE_INNER = "left_eye_inner"
    LEFT_EYE = "left_eye"
    LEFT_EYE_OUTER = "left_eye_outer"
    RIGHT_EYE_INNER = "right_eye_inner"
    RIGHT_EYE = "right_eye"
    RIGHT_EYE_OUTER = "right_eye_outer"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    MOUTH_LEFT = "mouth_left"
    MOUTH_RIGHT = "mouth_right"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_PINKY = "left_pinky"
    RIGHT_PINKY = "right_pinky"
    LEFT_INDEX = "left_index"
    RIGHT_INDEX = "right_index"
    LEFT_THUMB = "left_thumb"
    RIGHT_THUMB = "right_thumb"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"

class HandJointType(str, Enum):
    """Standardized hand joint names."""
    WRIST = "wrist"
    THUMB_CMC = "thumb_cmc"
    THUMB_MCP = "thumb_mcp"
    THUMB_IP = "thumb_ip"
    THUMB_TIP = "thumb_tip"
    INDEX_FINGER_MCP = "index_finger_mcp"
    INDEX_FINGER_PIP = "index_finger_pip"
    INDEX_FINGER_DIP = "index_finger_dip"
    INDEX_FINGER_TIP = "index_finger_tip"
    MIDDLE_FINGER_MCP = "middle_finger_mcp"
    MIDDLE_FINGER_PIP = "middle_finger_pip"
    MIDDLE_FINGER_DIP = "middle_finger_dip"
    MIDDLE_FINGER_TIP = "middle_finger_tip"
    RING_FINGER_MCP = "ring_finger_mcp"
    RING_FINGER_PIP = "ring_finger_pip"
    RING_FINGER_DIP = "ring_finger_dip"
    RING_FINGER_TIP = "ring_finger_tip"
    PINKY_MCP = "pinky_mcp"
    PINKY_PIP = "pinky_pip"
    PINKY_DIP = "pinky_dip"
    PINKY_TIP = "pinky_tip"

class Vector3D(BaseModel):
    """3D vector with confidence."""
    x: float = Field(..., ge=-10.0, le=10.0)  # Normalized coordinates
    y: float = Field(..., ge=-10.0, le=10.0)
    z: float = Field(..., ge=-10.0, le=10.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @classmethod
    def from_array(cls, array: np.ndarray, confidence: float = 1.0):
        """Create from numpy array."""
        return cls(x=float(array[0]), y=float(array[1]), z=float(array[2]), confidence=confidence)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

class HeadPose(BaseModel):
    """Head pose estimation."""
    yaw: float = Field(..., ge=-180.0, le=180.0)      # Rotation around Y-axis
    pitch: float = Field(..., ge=-90.0, le=90.0)      # Rotation around X-axis  
    roll: float = Field(..., ge=-180.0, le=180.0)     # Rotation around Z-axis
    confidence: float = Field(..., ge=0.0, le=1.0)

class ExpressionMetrics(BaseModel):
    """Facial expression metrics."""
    smile: float = Field(0.0, ge=0.0, le=1.0)
    frown: float = Field(0.0, ge=0.0, le=1.0)
    surprise: float = Field(0.0, ge=0.0, le=1.0)
    blink_left: float = Field(0.0, ge=0.0, le=1.0)
    blink_right: float = Field(0.0, ge=0.0, le=1.0)
    mouth_open: float = Field(0.0, ge=0.0, le=1.0)
    eyebrows_raised: float = Field(0.0, ge=0.0, le=1.0)

class BodyJoints(BaseModel):
    """Complete body joints representation."""
    # Core body joints
    nose: Optional[Vector3D] = None
    left_shoulder: Optional[Vector3D] = None
    right_shoulder: Optional[Vector3D] = None
    left_elbow: Optional[Vector3D] = None
    right_elbow: Optional[Vector3D] = None
    left_wrist: Optional[Vector3D] = None
    right_wrist: Optional[Vector3D] = None
    left_hip: Optional[Vector3D] = None
    right_hip: Optional[Vector3D] = None
    left_knee: Optional[Vector3D] = None
    right_knee: Optional[Vector3D] = None
    left_ankle: Optional[Vector3D] = None
    right_ankle: Optional[Vector3D] = None
    
    # Additional joints for completeness
    left_eye: Optional[Vector3D] = None
    right_eye: Optional[Vector3D] = None
    left_ear: Optional[Vector3D] = None
    right_ear: Optional[Vector3D] = None
    mouth_left: Optional[Vector3D] = None
    mouth_right: Optional[Vector3D] = None
    
    # Custom method to get all joints as dictionary
    def get_joints_dict(self) -> Dict[str, Vector3D]:
        """Get all joints as dictionary."""
        joints = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None and isinstance(field_value, Vector3D):
                joints[field_name] = field_value
        return joints

class HandJoints(BaseModel):
    """Complete hand joints representation."""
    wrist: Optional[Vector3D] = None
    thumb_cmc: Optional[Vector3D] = None
    thumb_mcp: Optional[Vector3D] = None
    thumb_ip: Optional[Vector3D] = None
    thumb_tip: Optional[Vector3D] = None
    index_finger_mcp: Optional[Vector3D] = None
    index_finger_pip: Optional[Vector3D] = None
    index_finger_dip: Optional[Vector3D] = None
    index_finger_tip: Optional[Vector3D] = None
    middle_finger_mcp: Optional[Vector3D] = None
    middle_finger_pip: Optional[Vector3D] = None
    middle_finger_dip: Optional[Vector3D] = None
    middle_finger_tip: Optional[Vector3D] = None
    ring_finger_mcp: Optional[Vector3D] = None
    ring_finger_pip: Optional[Vector3D] = None
    ring_finger_dip: Optional[Vector3D] = None
    ring_finger_tip: Optional[Vector3D] = None
    pinky_mcp: Optional[Vector3D] = None
    pinky_pip: Optional[Vector3D] = None
    pinky_dip: Optional[Vector3D] = None
    pinky_tip: Optional[Vector3D] = None

class FaceLandmarks(BaseModel):
    """Face landmarks representation."""
    landmarks: List[Vector3D] = Field(default_factory=list)  # 468 or 478 points
    expression: ExpressionMetrics = Field(default_factory=ExpressionMetrics)
    
    @validator('landmarks')
    def validate_landmarks_count(cls, v):
        """Validate landmarks count."""
        if len(v) not in [0, 468, 478]:
            logger.warning(f"Unexpected number of face landmarks: {len(v)}")
        return v

class SkeletonFrame(BaseModel):
    """
    Unified skeleton frame representation.
    Contains complete pose data for a single person in a single frame.
    """
    # Identification
    frame_id: int = Field(..., ge=0)
    timestamp: float = Field(..., ge=0.0)
    person_id: str = Field(..., pattern=r'^P\d+$')  # P1, P2, etc.
    source_video: Optional[str] = None
    
    # Pose data
    body: BodyJoints = Field(default_factory=BodyJoints)
    left_hand: Optional[HandJoints] = None
    right_hand: Optional[HandJoints] = None
    face: Optional[FaceLandmarks] = None
    head_pose: Optional[HeadPose] = None
    
    # Quality metrics
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    tracking_quality: float = Field(1.0, ge=0.0, le=1.0)
    
    # Additional metadata
    camera_id: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None  # (width, height)
    processing_time: Optional[float] = None
    
    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "person_id": self.person_id,
            "source_video": self.source_video,
            "body": {k: v.dict() if v else None for k, v in self.body.get_joints_dict().items()},
            "left_hand": self.left_hand.dict() if self.left_hand else None,
            "right_hand": self.right_hand.dict() if self.right_hand else None,
            "face": self.face.dict() if self.face else None,
            "head_pose": self.head_pose.dict() if self.head_pose else None,
            "overall_confidence": self.overall_confidence,
            "tracking_quality": self.tracking_quality,
            "camera_id": self.camera_id,
            "resolution": self.resolution,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_mediapipe(cls, 
                      frame_id: int,
                      timestamp: float,
                      person_id: str,
                      pose_landmarks: Any,
                      left_hand_landmarks: Any = None,
                      right_hand_landmarks: Any = None,
                      face_landmarks: Any = None,
                      pose_world_landmarks: Any = None) -> 'SkeletonFrame':
        """
        Create from MediaPipe results.
        
        Args:
            frame_id: Frame number
            timestamp: Timestamp in seconds
            person_id: Person identifier
            pose_landmarks: MediaPipe pose landmarks
            left_hand_landmarks: MediaPipe left hand landmarks
            right_hand_landmarks: MediaPipe right hand landmarks
            face_landmarks: MediaPipe face landmarks
            pose_world_landmarks: MediaPipe world landmarks
            
        Returns:
            SkeletonFrame instance
        """
        # Convert MediaPipe landmarks to our format
        body_joints = cls._convert_mediapipe_pose(pose_landmarks, pose_world_landmarks)
        left_hand = cls._convert_mediapipe_hand(left_hand_landmarks) if left_hand_landmarks else None
        right_hand = cls._convert_mediapipe_hand(right_hand_landmarks) if right_hand_landmarks else None
        face = cls._convert_mediapipe_face(face_landmarks) if face_landmarks else None
        
        return cls(
            frame_id=frame_id,
            timestamp=timestamp,
            person_id=person_id,
            body=body_joints,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
            head_pose=cls._estimate_head_pose(pose_landmarks)
        )
    
    @staticmethod
    def _convert_mediapipe_pose(landmarks, world_landmarks=None) -> BodyJoints:
        """Convert MediaPipe pose landmarks to BodyJoints."""
        # Implementation would map MediaPipe indices to our joint names
        # This is a simplified version
        body = BodyJoints()
        
        if landmarks:
            # Map MediaPipe indices to our joint names
            joint_mapping = {
                0: "nose",
                11: "left_shoulder",
                12: "right_shoulder",
                13: "left_elbow",
                14: "right_elbow",
                15: "left_wrist",
                16: "right_wrist",
                23: "left_hip",
                24: "right_hip",
                25: "left_knee",
                26: "right_knee",
                27: "left_ankle",
                28: "right_ankle"
            }
            
            for mp_idx, joint_name in joint_mapping.items():
                if mp_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[mp_idx]
                    setattr(body, joint_name, Vector3D(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        confidence=landmark.visibility
                    ))
        
        return body
    
    @staticmethod
    def _convert_mediapipe_hand(landmarks) -> HandJoints:
        """Convert MediaPipe hand landmarks to HandJoints."""
        hand = HandJoints()
        
        if landmarks and len(landmarks.landmark) >= 21:
            # Map MediaPipe hand indices to our joint names
            for i, landmark in enumerate(landmarks.landmark):
                joint_name = HandJointType.__members__.get(f"_{i}", None)
                if joint_name:
                    setattr(hand, joint_name, Vector3D(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        confidence=1.0  # MediaPipe doesn't provide hand confidence
                    ))
        
        return hand
    
    @staticmethod
    def _convert_mediapipe_face(landmarks) -> FaceLandmarks:
        """Convert MediaPipe face landmarks to FaceLandmarks."""
        face_landmarks = []
        
        if landmarks:
            for landmark in landmarks.landmark:
                face_landmarks.append(Vector3D(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    confidence=1.0  # MediaPipe doesn't provide face confidence
                ))
        
        return FaceLandmarks(landmarks=face_landmarks)
    
    @staticmethod
    def _estimate_head_pose(pose_landmarks) -> Optional[HeadPose]:
        """Estimate head pose from body landmarks."""
        if not pose_landmarks or len(pose_landmarks.landmark) < 33:
            return None
        
        # Simple head pose estimation using nose and eyes
        # This is a simplified version - in production, use proper head pose estimation
        try:
            nose = pose_landmarks.landmark[0]
            left_eye = pose_landmarks.landmark[2]
            right_eye = pose_landmarks.landmark[5]
            
            # Calculate simple head orientation
            # This is very basic - consider using solvePnP for proper estimation
            eye_vector = np.array([right_eye.x - left_eye.x, 
                                 right_eye.y - left_eye.y,
                                 right_eye.z - left_eye.z])
            
            # Simple approximation
            yaw = np.arctan2(eye_vector[0], eye_vector[2])
            pitch = np.arctan2(eye_vector[1], eye_vector[2])
            
            return HeadPose(
                yaw=np.degrees(yaw),
                pitch=np.degrees(pitch),
                roll=0.0,  # Would need more points for roll
                confidence=min(nose.visibility, left_eye.visibility, right_eye.visibility)
            )
        except Exception as e:
            logger.warning(f"Head pose estimation failed: {e}")
            return None
