"""
Verification script for the Feature Extraction Layer.
"""
import numpy as np
from core.perception.feature_extraction import FeatureExtractor, PoseFeatures
from loguru import logger

def test_feature_extraction():
    extractor = FeatureExtractor(window_size=5)
    
    # Mock pose data for frame 0
    pose_data_f0 = {
        "body_landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0}] * 33,
        "left_hand_landmarks": [{"x": 0.4, "y": 0.4, "z": 0.0}] * 21,
        "right_hand_landmarks": [{"x": 0.6, "y": 0.6, "z": 0.0}] * 21,
        "face_landmarks": [{"x": 0.5, "y": 0.45, "z": 0.0}] * 468
    }
    
    # Process frame 0
    logger.info("Processing Frame 0...")
    features_f0 = extractor.extract_features(pose_data_f0, track_id=1, frame_idx=0)
    
    assert len(features_f0.velocities) == 0, "Velocities should be empty for the first frame"
    
    # Mock pose data for frame 1 (moved slightly)
    # Move body center from 0.5 to 0.51 in 1/30th of a second
    # dx = 0.01. dt = 1/30. velocity = 0.01 / (1/30) = 0.3
    pose_data_f1 = {
        "body_landmarks": [{"x": 0.51, "y": 0.5, "z": 0.0}] * 33,
        "left_hand_landmarks": [{"x": 0.4, "y": 0.4, "z": 0.0}] * 21,
        "right_hand_landmarks": [{"x": 0.6, "y": 0.6, "z": 0.0}] * 21,
        "face_landmarks": [{"x": 0.5, "y": 0.45, "z": 0.0}] * 468
    }
    
    logger.info("Processing Frame 1...")
    features_f1 = extractor.extract_features(pose_data_f1, track_id=1, frame_idx=1)
    
    # Check velocity for body landmarks
    if "body_landmarks" in features_f1.velocities:
        avg_velocity = np.mean(features_f1.velocities["body_landmarks"])
        logger.info(f"Average Body Velocity: {avg_velocity:.4f}")
        assert np.isclose(avg_velocity, 0.3, atol=0.01), f"Expected velocity 0.3, got {avg_velocity}"
    else:
        raise AssertionError("Body landmarks velocity not calculated")

    # Check stability score
    logger.info(f"Body Stability Score: {features_f1.stability_scores.get('body_landmarks')}")
    
    logger.success("Feature Extraction Layer verification successful!")

if __name__ == "__main__":
    try:
        test_feature_extraction()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        exit(1)
