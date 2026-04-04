"""
Holistic pose estimation combining body, hands, and face.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
import mediapipe as mp
from loguru import logger

from ..pose_base import BasePoseEstimator, PoseResult
from .body_pose import BodyPoseEstimator
from .hand_pose import HandPoseEstimator
from .face_mesh import FaceMeshEstimator

class HolisticPoseEstimator(BasePoseEstimator):
    """Unified holistic pose estimation pipeline."""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        super().__init__(min_detection_confidence)
        self.model_complexity = model_complexity
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize individual estimators
        self.body_estimator = BodyPoseEstimator(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.hand_estimator = HandPoseEstimator(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.face_estimator = FaceMeshEstimator(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = None
    
    def initialize(self):
        """Initialize all estimators."""
        try:
            # Initialize MediaPipe Holistic directly for better integration
            self.holistic = self.mp_holistic.Holistic(
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=True
            )
            
            # Also initialize individual estimators
            self.body_estimator.initialize()
            self.hand_estimator.initialize()
            self.face_estimator.initialize()
            
            self.initialized = True
            logger.success("Initialized Holistic Pose estimator")
        except Exception as e:
            logger.error(f"Failed to initialize Holistic Pose: {e}")
            raise
    
    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> PoseResult:
        """Estimate holistic pose."""
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
            
            # Process with holistic model
            results = self.holistic.process(rgb_frame)
            
            # Extract all landmarks
            landmarks = {}
            confidence = {}
            world_landmarks = {}
            
            # Body landmarks
            if results.pose_landmarks:
                body_2d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                landmarks['body'] = body_2d
                confidence['body'] = np.array([lm.visibility for lm in results.pose_landmarks.landmark])
            
            if results.pose_world_landmarks:
                body_3d = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
                world_landmarks['body'] = body_3d
            
            # Hand landmarks
            if results.left_hand_landmarks:
                left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                landmarks['left_hand'] = left_hand
                confidence['left_hand'] = np.ones(21)  # MediaPipe doesn't provide hand confidence
            
            if results.right_hand_landmarks:
                right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
                landmarks['right_hand'] = right_hand
                confidence['right_hand'] = np.ones(21)
            
            # Face landmarks
            if results.face_landmarks:
                face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
                landmarks['face'] = face
                confidence['face'] = np.ones(len(results.face_landmarks.landmark))
            
            result = PoseResult(
                landmarks=landmarks,
                confidence=confidence,
                world_landmarks=world_landmarks if world_landmarks else None
            )
            
            return self.filter_by_confidence(result)
            
        except Exception as e:
            logger.error(f"Holistic pose estimation failed: {e}")
            return PoseResult({}, {})
