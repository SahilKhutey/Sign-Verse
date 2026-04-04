"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, HttpUrl, conlist
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

class UploadSource(str, Enum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"
    EXTERNAL = "external"

class VideoUploadRequest(BaseModel):
    source: UploadSource = UploadSource.UPLOAD
    filename: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class VideoUploadResponse(BaseModel):
    id: str
    filename: str
    source: UploadSource
    upload_time: datetime
    status: str
    message: Optional[str] = None
    file_path: Optional[str] = None

class PoseEstimationRequest(BaseModel):
    video_id: str
    model_name: str = "mediapipe"
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    include_3d: bool = False
    smooth_output: bool = True

class PoseEstimationResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    total_frames: int
    processed_frames: int
    estimated_time: Optional[float] = None
    result_path: Optional[str] = None

class Keypoint3D(BaseModel):
    x: float
    y: float
    z: float
    confidence: float = Field(..., ge=0, le=1)
    visible: bool = True

class PoseFrame(BaseModel):
    frame_number: int
    timestamp: float
    keypoints: List[Keypoint3D]
    bounding_box: Optional[Dict[str, float]] = None

class PoseResult(BaseModel):
    video_id: str
    frames: List[PoseFrame]
    model_name: str
    processing_time: float
    resolution: Optional[Tuple[int, int]] = None

class SimulationRequest(BaseModel):
    pose_data_id: str
    character_type: str = "humanoid"
    output_format: str = "fbx"
    include_constraints: bool = True
    max_velocity: Optional[float] = None

class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    output_formats: List[str]
    file_paths: Dict[str, str]
    processing_time: float
    constraints_violations: Optional[int] = None

class ErrorResponse(BaseModel):
    error: str
    code: int
    message: str
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    requests_processed: int
    services: Dict[str, str]

class BatchRequest(BaseModel):
    items: conlist(PoseEstimationRequest, min_items=1, max_items=10)

class BatchResponse(BaseModel):
    processed: int
    successful: int
    failed: int
    results: List[PoseEstimationResponse]
    errors: List[ErrorResponse]
