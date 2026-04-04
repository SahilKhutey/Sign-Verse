"""
Test script for the refactored HolisticPoseEstimator (Tasks API).
"""
import cv2
import numpy as np
from core.perception.pose_estimation import HolisticPoseEstimator
from loguru import logger
import os

def test_pose_estimation():
    # Check if models exist
    model_dir = "models/perception"
    required_models = ["pose_landmarker.task", "hand_landmarker.task", "face_landmarker.task"]
    
    missing = [m for m in required_models if not os.path.exists(os.path.join(model_dir, m))]
    if missing:
        logger.warning(f"Missing models: {missing}. Test cannot run yet.")
        return

    estimator = HolisticPoseEstimator(model_path=model_dir)
    
    # Create a dummy image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a line so it's not totally black (though MediaPipe might still fail to find a person)
    cv2.line(frame, (0, 0), (640, 480), (255, 255, 255), 5)
    
    logger.info("Processing blank frame...")
    result = estimator.process_frame(frame)
    
    logger.info(f"Body Landmarks: {len(result.body_landmarks) if result.body_landmarks else 'None'}")
    logger.info(f"Face Landmarks: {len(result.face_landmarks) if result.face_landmarks else 'None'}")
    logger.info(f"Left Hand: {len(result.left_hand_landmarks) if result.left_hand_landmarks else 'None'}")
    logger.info(f"Right Hand: {len(result.right_hand_landmarks) if result.right_hand_landmarks else 'None'}")
    
    estimator.close()
    logger.success("Pose Estimation Layer (Tasks API) initialized correctly!")

if __name__ == "__main__":
    test_pose_estimation()
