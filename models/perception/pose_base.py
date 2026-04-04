"""
Base class for all pose estimation models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
from loguru import logger

@dataclass
class PoseResult:
    landmarks: Dict[str, np.ndarray]  # landmark_type -> [N, 3] array
    confidence: Dict[str, np.ndarray]  # landmark_type -> [N] array
    world_landmarks: Optional[Dict[str, np.ndarray]] = None
    segmentation: Optional[np.ndarray] = None
    additional_data: Dict[str, Any] = None

class BasePoseEstimator(ABC):
    """Abstract base class for pose estimators."""
    
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.initialized = False
    
    @abstractmethod
    def initialize(self):
        """Initialize the pose estimator."""
        pass
    
    @abstractmethod
    def estimate(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> PoseResult:
        """
        Estimate pose in a single frame.
        
        Args:
            frame: Input frame
            bbox: Optional bounding box for ROI
            
        Returns:
            PoseResult object
        """
        pass
    
    def estimate_batch(self, frames: List[np.ndarray], bboxes: Optional[List[np.ndarray]] = None) -> List[PoseResult]:
        """
        Estimate pose in multiple frames.
        Default implementation loops over single-frame estimation.
        
        Args:
            frames: List of input frames
            bboxes: Optional list of bounding boxes
            
        Returns:
            List of PoseResult objects
        """
        if bboxes is None:
            bboxes = [None] * len(frames)
        
        return [self.estimate(frame, bbox) for frame, bbox in zip(frames, bboxes)]
    
    def filter_by_confidence(self, result: PoseResult) -> PoseResult:
        """Filter landmarks by confidence threshold."""
        filtered_landmarks = {}
        filtered_confidence = {}
        
        for landmark_type, landmarks in result.landmarks.items():
            confidences = result.confidence.get(landmark_type, np.ones(len(landmarks)))
            mask = confidences >= self.min_confidence
            
            filtered_landmarks[landmark_type] = landmarks[mask]
            filtered_confidence[landmark_type] = confidences[mask]
        
        return PoseResult(
            landmarks=filtered_landmarks,
            confidence=filtered_confidence,
            world_landmarks=result.world_landmarks,
            segmentation=result.segmentation,
            additional_data=result.additional_data
        )
    
    def __call__(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> PoseResult:
        """Convenience method for pose estimation."""
        return self.estimate(frame, bbox)
