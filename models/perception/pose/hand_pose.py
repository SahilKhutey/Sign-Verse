"""
Hand pose estimation using MediaPipe Hands.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
import mediapipe as mp
from loguru import logger

from ..pose_base import BasePoseEstimator, PoseResult

class HandPoseEstimator(BasePoseEstimator):
    """MediaPipe Hands for hand pose estimation."""
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        super().__init__(min_detection_confidence)
        self.max_num_hands = max_num_hands
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
    
    def initialize(self):
        """Initialize MediaPipe Hands."""
        try:
            self.hands = self.mp_hands.Hands(
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.initialized = True
            logger.success("Initialized MediaPipe Hands estimator")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Hands: {e}")
            raise
    
    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> PoseResult:
        """Estimate hand pose."""
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
            results = self.hands.process(rgb_frame)
            
            # Extract landmarks
            landmarks = {}
            confidence = {}
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_type = "left" if results.multi_handedness[hand_idx].classification[0].label == "Left" else "right"
                    
                    # Extract 2D landmarks (21 points per hand)
                    hand_points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    landmarks[f'hand_{hand_type}'] = hand_points
                    
                    # Use presence for confidence (MediaPipe Hands doesn't provide visibility)
                    conf = np.ones(21) * results.multi_handedness[hand_idx].classification[0].score
                    confidence[f'hand_{hand_type}'] = conf
            
            result = PoseResult(landmarks=landmarks, confidence=confidence)
            return self.filter_by_confidence(result)
            
        except Exception as e:
            logger.error(f"Hand pose estimation failed: {e}")
            return PoseResult({}, {})
