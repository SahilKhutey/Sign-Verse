import pytest
import numpy as np
from models.perception.features.gestures import is_open_palm, is_fist
from models.perception.features.expressions import is_smiling
from models.perception.temporal.smoothing import TemporalSmoother

def test_is_open_palm():
    # Mock open hand: [21, 3] array
    # Wrist at index 0: [0, 0, 0]
    # MCPs at 5, 9, 13, 17: [0.1, 0, 0]
    # Tips at 8, 12, 16, 20: [0.2, 0, 0]
    landmarks = np.zeros((21, 3))
    landmarks[[5, 9, 13, 17]] = [0.1, 0, 0]
    landmarks[[8, 12, 16, 20]] = [0.2, 0, 0]
    
    assert is_open_palm(landmarks) == True

def test_is_fist():
    # Mock fist: tips closer to wrist than MCPs
    landmarks = np.zeros((21, 3))
    landmarks[[5, 9, 13, 17]] = [0.2, 0, 0]
    landmarks[[8, 12, 16, 20]] = [0.1, 0, 0]
    
    assert is_fist(landmarks) == True

def test_is_smiling():
    # Face landmarks [468, 3]
    # Mouth Mesh: 61, 291 (width), 13, 14 (height)
    landmarks = np.zeros((468, 3))
    landmarks[61] = [-0.1, 0, 0]
    landmarks[291] = [0.1, 0, 0]
    landmarks[13] = [0, 0.01, 0]
    landmarks[14] = [0, -0.01, 0]
    
    assert is_smiling(landmarks) == True

def test_temporal_smoothing():
    smoother = TemporalSmoother(window_size=3)
    landmarks = {
        "body": np.array([[1.0, 1.0, 0.0]] * 33)
    }
    
    # Send 3 frames with same data
    for i in range(3):
        smoothed = smoother.smooth(track_id=1, current_landmarks=landmarks)
        
    assert smoothed["body"][0][0] == pytest.approx(1.0)
    
    # Send jumpy data
    landmarks_noisy = {
        "body": np.array([[2.0, 2.0, 0.0]] * 33)
    }
    smoothed = smoother.smooth(track_id=1, current_landmarks=landmarks_noisy)
    # History: [1, 1, 2] -> Mean = 1.33
    assert smoothed["body"][0][0] == pytest.approx(1.33, rel=1e-2)
