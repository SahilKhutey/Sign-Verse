"""
Pydantic models for data layer entities.
Ensures consistency across the entire system.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pathlib import Path

class DataSource(str, Enum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"
    SYNTHETIC = "synthetic"
    EXTERNAL = "external"

class Keypoint(BaseModel):
    id: int
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = Field(..., ge=0, le=1)
    visible: bool = True

    @validator('name')
    def name_must_be_valid(cls, v):
        from configs import get_config
        config = get_config()
        valid_names = config.keypoint_labels.labels
        if v not in valid_names:
            raise ValueError(f"Keypoint name '{v}' not in valid labels: {valid_names}")
        return v

class PoseData(BaseModel):
    """Standardized pose data model."""
    version: str = "1.0.0"
    source_video: str
    frame_number: int
    timestamp: float
    camera_id: Optional[str] = None
    subject_id: Optional[str] = None
    keypoints: List[Keypoint]
    bounding_box: Optional[Dict[str, float]] = None

    class Config:
        schema_extra = {
            "example": {
                "version": "1.0.0",
                "source_video": "upload_abc123_20231015.mp4",
                "frame_number": 42,
                "timestamp": 3.14,
                "keypoints": [
                    {"id": 0, "name": "nose", "x": 0.5, "y": 0.2, "confidence": 0.9},
                    {"id": 1, "name": "left_eye", "x": 0.4, "y": 0.2, "confidence": 0.8}
                ]
            }
        }

class DatasetManifest(BaseModel):
    """Metadata for a versioned dataset."""
    name: str
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = None
    source_data: List[str] = Field(..., description="Paths to source videos")
    statistics: Dict[str, Any] = Field(default_factory=dict)
    splits: Dict[str, int] = Field(..., description="Number of samples per split")

    @validator('version')
    def version_format(cls, v):
        # Basic semantic versioning check
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError('Version must be in format X.Y.Z')
        return v
