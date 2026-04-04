"""
Pipeline tests for SignVerse 3D Perception and Pose Estimation.
Mocks heavy AI model loads to ensure rapid test execution.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Robust mocking: Mock heavy dependencies before any server/pipeline imports
# This prevents the system from attempting to load GPU-bound or missing libraries
with patch('models.perception.detection.yolo_detector.YOLODetector'), \
     patch('models.perception.tracking.bytetrack.ByteTracker'), \
     patch('models.perception.pose.holistic.HolisticPoseEstimator'):
    from core.perception.pipeline import PerceptionPipeline
from core.data_models import SkeletonFrame

class TestPoseEstimationPipeline:
    """Test the pose estimation component within the perception system."""
    
    @pytest.fixture
    def mock_frame(self):
        """Create a mock video frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_perception_processing_loop(self, mock_frame):
        """Test the perception processing loop with mocked ML components."""
        # Initialize system (with mocked dependencies)
        system = PerceptionPipeline()
        
        # Mock detector output
        # Using a mock detection object that matches YOLODetector output
        mock_detection = Mock()
        mock_detection.bbox = [100, 100, 200, 200]
        mock_detection.confidence = 0.9
        mock_detection.class_name = "person"
        system.detector.detect.return_value = [mock_detection]
        
        # Mock tracker output
        mock_track = Mock()
        mock_track.track_id = 1
        mock_track.bbox = [100, 100, 200, 200]
        system.tracker.update.return_value = [mock_track]
        
        # Mock pose estimator output
        # Using a mock that matches HolisticPoseEstimator.estimate() -> PoseResult
        from models.perception.pose_base import PoseResult
        mock_pose = PoseResult(
            landmarks={"body": np.zeros((33, 3))},
            confidence={"body": np.ones(33)}
        )
        system.pose_estimator.estimate.return_value = mock_pose
        
        # Process frame
        results = system.process_frame(mock_frame)
        
        # Verify processing results
        assert 1 in results # Track ID 1
        assert "pose" in results[1]
        assert results[1]["pose"].landmarks is not None
