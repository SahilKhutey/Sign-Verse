"""
Database storage schema for pose data and trajectories.
"""
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from .skeleton import SkeletonFrame
    from .trajectory import PersonTrajectory

class StorageFormat(str, Enum):
    """Supported storage formats."""
    JSON = "json"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    TF_RECORD = "tf_record"

class PoseStorageSchema(BaseModel):
    """Schema for storing pose data in databases."""
    id: Optional[str] = None
    frame_id: int
    timestamp: float
    person_id: str
    source_video: str
    camera_id: Optional[str] = None
    
    # Pose data (normalized)
    body_data: Dict[str, Any]  # JSON-serialized body joints
    left_hand_data: Optional[Dict[str, Any]] = None
    right_hand_data: Optional[Dict[str, Any]] = None
    face_data: Optional[Dict[str, Any]] = None
    head_pose_data: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    overall_confidence: float
    tracking_quality: float
    
    # Processing info
    processing_time: float
    model_version: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Additional metadata
    resolution: Optional[Tuple[int, int]] = None
    frame_rate: Optional[float] = None
    environment: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @classmethod
    def from_skeleton_frame(cls, frame: 'SkeletonFrame', model_version: str = "1.0.0") -> 'PoseStorageSchema':
        """Create from SkeletonFrame."""
        return cls(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            person_id=frame.person_id,
            source_video=frame.source_video or "unknown",
            camera_id=frame.camera_id,
            body_data=frame.body.dict(),
            left_hand_data=frame.left_hand.dict() if frame.left_hand else None,
            right_hand_data=frame.right_hand.dict() if frame.right_hand else None,
            face_data=frame.face.dict() if frame.face else None,
            head_pose_data=frame.head_pose.dict() if frame.head_pose else None,
            overall_confidence=frame.overall_confidence,
            tracking_quality=frame.tracking_quality,
            processing_time=frame.processing_time or 0.0,
            model_version=model_version,
            resolution=frame.resolution,
            frame_rate=30.0  # Default assumption
        )

class TrajectoryStorageSchema(BaseModel):
    """Schema for storing trajectories."""
    id: Optional[str] = None
    person_id: str
    start_frame: int
    end_frame: int
    duration: float
    frame_count: int
    
    # Trajectory data
    trajectory_data: Dict[str, Any]  # JSON-serialized trajectory
    joint_list: List[str]
    
    # Statistics
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    avg_velocity: Optional[Dict[str, float]] = None
    avg_acceleration: Optional[Dict[str, float]] = None
    
    # Metadata
    source_video: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @classmethod
    def from_person_trajectory(cls, trajectory: 'PersonTrajectory', 
                              source_video: str = "unknown") -> 'TrajectoryStorageSchema':
        """Create from PersonTrajectory."""
        # Calculate statistics
        confidences = [point.confidence for point in trajectory.trajectory]
        
        return cls(
            person_id=trajectory.person_id,
            start_frame=trajectory.start_frame,
            end_frame=trajectory.end_frame,
            duration=trajectory.trajectory[-1].timestamp - trajectory.trajectory[0].timestamp if trajectory.trajectory else 0.0,
            frame_count=len(trajectory.trajectory),
            trajectory_data={
                "points": [point.__dict__ for point in trajectory.trajectory]
            },
            joint_list=list(trajectory.trajectory[0].joints.keys()) if trajectory.trajectory else [],
            avg_confidence=float(np.mean(confidences)) if confidences else 0.0,
            min_confidence=float(np.min(confidences)) if confidences else 0.0,
            max_confidence=float(np.max(confidences)) if confidences else 0.0,
            source_video=source_video
        )

class DatasetMetadata(BaseModel):
    """Metadata for pose datasets."""
    name: str
    version: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Statistics
    total_frames: int
    total_persons: int
    total_trajectories: int
    avg_frames_per_trajectory: float
    duration: float  # seconds
    
    # Data characteristics
    resolution: Tuple[int, int]
    frame_rate: float
    joint_types: List[str]
    has_hands: bool
    has_face: bool
    has_head_pose: bool
    
    # Source information
    source_videos: List[str]
    camera_configs: Optional[Dict[str, Any]] = None
    
    # Processing info
    model_versions: Dict[str, str]
    processing_parameters: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
