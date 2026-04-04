"""
Pipeline tests for pose estimation.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from models.inference.predict_pose import PosePredictor
from core.data_models import PoseData, Keypoint

class TestPoseEstimation:
    """Test pose estimation pipeline."""
    
    @pytest.fixture
    def mock_pose_data(self):
        """Create mock pose data for testing."""
        keypoints = [
            Keypoint(
                id=i,
                name=f"joint_{i}",
                x=np.random.random(),
                y=np.random.random(),
                z=np.random.random(),
                confidence=0.8 + 0.2 * np.random.random(),
                visible=True
            )
            for i in range(17)  # 17 keypoints like COCO format
        ]
        
        return PoseData(
            source_video="test_video.mp4",
            frame_number=1,
            timestamp=1.0,
            keypoints=keypoints
        )
    
    @patch('models.inference.predict_pose.PosePredictor._load_model')
    def test_pose_prediction(self, mock_load_model, mock_pose_data):
        """Test pose prediction functionality."""
        # Mock the model loading
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Mock model prediction
        mock_output = Mock()
        mock_output.squeeze.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.random.random((1, 51))
        mock_model.return_value = mock_output
        
        # Initialize predictor
        predictor = PosePredictor()
        
        # Test prediction
        result = predictor.predict([mock_pose_data])
        
        assert len(result) == 1
        assert isinstance(result[0], PoseData)
        assert len(result[0].keypoints) == len(mock_pose_data.keypoints)
