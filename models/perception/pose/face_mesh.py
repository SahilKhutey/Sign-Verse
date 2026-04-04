"""
Face mesh estimation using MediaPipe Face Mesh.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
import mediapipe as mp
from loguru import logger

from ..pose_base import BasePoseEstimator, PoseResult

class FaceMeshEstimator(BasePoseEstimator):
    """MediaPipe Face Mesh for facial landmark estimation."""
    
    def __init__(self, 
                 max_num_faces: int = 1,
                 refine_landmarks: bool = False,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        super().__init__(min_detection_confidence)
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_tracking_confidence = min_tracking_confidence
        self.face_mesh = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
    
    def initialize(self):
        """Initialize MediaPipe Face Mesh."""
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.initialized = True
            logger.success("Initialized MediaPipe Face Mesh estimator")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}")
            raise
    
    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> PoseResult:
        """Estimate face mesh."""
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
            results = self.face_mesh.process(rgb_frame)
            
            # Extract landmarks
            landmarks = {}
            confidence = {}
            
            if results.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Extract 2D landmarks (468 or 478 points with refinement)
                    face_points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
                    landmarks[f'face_{face_idx}'] = face_points
                    
                    # Use presence for confidence
                    conf = np.ones(len(face_landmarks.landmark))
                    confidence[f'face_{face_idx}'] = conf
            
            result = PoseResult(landmarks=landmarks, confidence=confidence)
            return self.filter_by_confidence(result)
            
        except Exception as e:
            logger.error(f"Face mesh estimation failed: {e}")
            return PoseResult({}, {})
