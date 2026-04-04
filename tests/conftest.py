"""
Pytest configuration and fixtures for SignVerse tests.
Standardized setup for data models and storage.
"""
import pytest
import sys
import numpy as np
from unittest.mock import Mock, patch

# CRITICAL: Mock ONLY heavy AI/Network dependencies to avoid library-induced errors.
# We avoid mocking standard utility libraries like loguru or cv2 unless absolutely necessary.
MOCK_MODULES = [
    'vllm',
    'mediapipe',
    'mediapipe.solutions',
    'mediapipe.solutions.holistic',
    'ultralytics'
]

for module in MOCK_MODULES:
    if module not in sys.modules:
        m = Mock()
        if module == 'mediapipe':
            m.solutions.holistic.Holistic = Mock()
        sys.modules[module] = m

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for all tests."""
    with patch('redis.Redis'), \
         patch('requests.get'), \
         patch('yt_dlp.YoutubeDL'):
        yield

@pytest.fixture
def mock_skeleton_frame():
    """Provide a modern mock skeleton frame for testing."""
    from core.data_models import SkeletonFrame, BodyJoints, Vector3D
    
    body = BodyJoints(
        nose=Vector3D(x=0.5, y=0.5, z=0.0, confidence=0.9),
        left_shoulder=Vector3D(x=0.4, y=0.6, z=0.0, confidence=0.8)
    )
    
    return SkeletonFrame(
        person_id="P1",
        frame_id=1,
        timestamp=0.033,
        body=body
    )
