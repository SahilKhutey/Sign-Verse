"""
Pytest configuration and fixtures for SignVerse tests.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for all tests."""
    # Mock external dependencies
    with patch('redis.Redis'), \
         patch('requests.get'), \
         patch('yt_dlp.YoutubeDL'):
        yield

@pytest.fixture
def sample_video_path():
    """Provide a sample video path for testing."""
    # In real implementation, this would be an actual test video
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        yield f.name

@pytest.fixture
def mock_pose_data():
    """Provide mock pose data for testing."""
    from core.data_models import PoseData, Keypoint
    
    keypoints = [
        Keypoint(
            id=i,
            name=f"joint_{i}",
            x=0.5,
            y=0.5,
            z=0.0,
            confidence=0.9,
            visible=True
        )
        for i in range(5)
    ]
    
    return PoseData(
        source_video="test_video.mp4",
        frame_number=1,
        timestamp=1.0,
        keypoints=keypoints
    )
