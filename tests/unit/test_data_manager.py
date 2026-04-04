"""
Unit tests for data management utilities.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from core.data_manager import DataManager, DataSource
from core.data_models import PoseData, Keypoint

class TestDataManager:
    """Test data manager functionality."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def data_manager(self, temp_data_dir):
        """Create DataManager instance with temp directory."""
        return DataManager(Path(temp_data_dir))
    
    def test_directory_creation(self, data_manager, temp_data_dir):
        """Test that directories are created correctly."""
        data_manager._ensure_directories()
        
        expected_dirs = [
            "raw/uploads",
            "raw/youtube_videos",
            "processed/frames",
            "processed/poses",
            "processed/annotations",
            "labeled/actions",
            "labeled/joints",
            "datasets",
            "metadata"
        ]
        
        for rel_dir in expected_dirs:
            full_path = Path(temp_data_dir) / rel_dir
            assert full_path.exists(), f"Directory {rel_dir} should exist"
            assert full_path.is_dir(), f"Path {rel_dir} should be a directory"
    
    def test_video_path_resolution(self, data_manager, temp_data_dir):
        """Test video path resolution."""
        # Test upload path
        upload_path = data_manager.get_video_path(DataSource.UPLOAD, "test.mp4")
        expected_path = Path(temp_data_dir) / "raw" / "uploads" / "test.mp4"
        assert upload_path == expected_path
        
        # Test YouTube path
        youtube_path = data_manager.get_video_path(DataSource.YOUTUBE, "video.mp4")
        expected_path = Path(temp_data_dir) / "raw" / "youtube_videos" / "video.mp4"
        assert youtube_path == expected_path
    
    def test_pose_data_saving(self, data_manager, temp_data_dir):
        """Test pose data saving and loading."""
        data_manager._ensure_directories()
        
        # Create test pose data
        keypoints = [
            Keypoint(
                id=0,
                name="nose",
                x=0.5,
                y=0.5,
                z=0.0,
                confidence=0.9,
                visible=True
            )
        ]
        
        pose_data = PoseData(
            source_video="test_video.mp4",
            frame_number=1,
            timestamp=1.0,
            keypoints=keypoints
        )
        
        # Save pose data
        output_path = data_manager.save_pose_data(pose_data, "test_pose.json")
        assert output_path.exists()
        
        # Load pose data
        loaded_data = data_manager.load_pose_data("test_pose.json")
        assert loaded_data.source_video == "test_video.mp4"
        assert len(loaded_data.keypoints) == 1
        assert loaded_data.keypoints[0].name == "nose"
