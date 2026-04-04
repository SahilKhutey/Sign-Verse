"""
Unit tests for data models and validation.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from core.data_models import PoseData, Keypoint, DatasetManifest

class TestDataModels:
    """Test data model validation."""
    
    def test_keypoint_validation(self):
        """Test Keypoint model validation."""
        # Valid keypoint
        valid_kp = Keypoint(
            id=0,
            name="nose",
            x=0.5,
            y=0.5,
            z=0.0,
            confidence=0.9,
            visible=True
        )
        assert valid_kp.id == 0
        assert valid_kp.confidence == 0.9
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            Keypoint(
                id=0,
                name="nose",
                x=0.5,
                y=0.5,
                confidence=1.5,  # > 1.0
                visible=True
            )
    
    def test_posedata_validation(self):
        """Test PoseData model validation."""
        keypoints = [
            Keypoint(
                id=i,
                name=f"joint_{i}",
                x=0.5,
                y=0.5,
                z=0.0,
                confidence=0.8,
                visible=True
            )
            for i in range(5)
        ]
        
        # Valid pose data
        pose_data = PoseData(
            source_video="test_video.mp4",
            frame_number=1,
            timestamp=1.0,
            keypoints=keypoints
        )
        assert pose_data.frame_number == 1
        assert len(pose_data.keypoints) == 5
        
        # Missing required fields
        with pytest.raises(ValidationError):
            PoseData(
                source_video="test_video.mp4",
                frame_number=1,
                # Missing timestamp and keypoints
            )
    
    def test_dataset_manifest_validation(self):
        """Test DatasetManifest validation."""
        # Valid dataset manifest
        manifest = DatasetManifest(
            name="test_dataset",
            version="1.0.0",
            source_data=["video1.mp4", "video2.mp4"],
            splits={"train": 100, "val": 20, "test": 30}
        )
        assert manifest.name == "test_dataset"
        assert manifest.version == "1.0.0"
        
        # Invalid version format
        with pytest.raises(ValidationError):
            DatasetManifest(
                name="test_dataset",
                version="invalid_ver",
                source_data=[],
                splits={}
            )
