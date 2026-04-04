"""
Base class for all detection models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from loguru import logger

@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    features: Any = None

class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.initialized = False
    
    @abstractmethod
    def initialize(self):
        """Initialize the detector model."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect objects in multiple frames (batch processing).
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists for each frame
        """
        pass
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess input frame for detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        return frame
    
    def postprocess(self, detections: List[Detection]) -> List[Detection]:
        """
        Postprocess detections (filtering, NMS, etc.).
        
        Args:
            detections: Raw detections
            
        Returns:
            Processed detections
        """
        # Filter by confidence
        filtered = [d for d in detections if d.confidence >= self.confidence_threshold]
        return filtered
    
    def __call__(self, frame: np.ndarray) -> List[Detection]:
        """Convenience method for detection."""
        return self.detect(frame)
