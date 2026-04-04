"""
Body pose estimation using MediaPipe Pose.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
import mediapipe as mp
from loguru import logger

from ..pose_base import BasePoseEstimator, PoseResult

class BodyPoseEstimator(BasePoseEstimator):
    """MediaPipe Pose for body pose estimation."""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        super().__init__(min_detection_confidence)
        self.model_complexity = model_complexity
        self.min_tracking_confidence = min_tracking_confidence
        self.pose = None
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
    
    def initialize(self):
        """Initialize MediaPipe Pose."""
        try:
            self.pose = self.mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=True
            )
            self.initialized = True
            logger.success("Initialized MediaPipe Pose estimator")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose: {e}")
            raise
    
    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> PoseResult:
        """Estimate body pose."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Extract ROI if bounding box provided
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    return PoseResult({}, {})
            else:
                roi = frame
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            # Extract landmarks
            landmarks = {}
            confidence = {}
            world_landmarks = {}
            
            if results.pose_landmarks:
                # Extract 2D landmarks (33 points)
                body_2d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                landmarks['body'] = body_2d
                confidence['body'] = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
            
            if results.pose_world_landmarks:
                # Extract 3D world landmarks
                body_3d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
                world_landmarks['body'] = body_3d
            
            result = PoseResult(
                landmarks=landmarks,
                confidence=confidence,
                world_landmarks=world_landmarks if world_landmarks else None
            )
            
            return self.filter_by_confidence(result)
            
        except Exception as e:
            logger.error(f"Body pose estimation failed: {e}")
            return PoseResult({}, {})
    
    def estimate_batch(self, frames: List[np.ndarray], bboxes: Optional[List[np.ndarray]] = None) -> List[PoseResult]:
        """Estimate pose for batch of frames."""
        results = []
        for i, frame in enumerate(frames):
            bbox = bboxes[i] if bboxes and i < len(bboxes) else None
            result = self.estimate(frame, bbox)
            results.append(result)
        return results
