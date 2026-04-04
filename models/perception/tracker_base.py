"""
Base class for all tracking models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from loguru import logger

@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    confidence: float
    features: Any
    frame_count: int
    lost_count: int
    history: List[Any] = None

class BaseTracker(ABC):
    """Abstract base class for object trackers."""
    
    def __init__(self, max_lost: int = 30):
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.initialized = False
    
    @abstractmethod
    def initialize(self):
        """Initialize the tracker."""
        pass
    
    @abstractmethod
    def update(self, detections: List[Any], frame_idx: int) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection objects
            frame_idx: Current frame index
            
        Returns:
            List of active tracks
        """
        pass
    
    @abstractmethod
    def predict(self) -> List[Track]:
        """
        Predict next state of all tracks.
        
        Returns:
            List of predicted tracks
        """
        pass
    
    def remove_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.lost_count > self.max_lost:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks (not lost)."""
        return [track for track in self.tracks.values() if track.lost_count == 0]
    
    def __call__(self, detections: List[Any], frame_idx: int) -> List[Track]:
        """Convenience method for tracking update."""
        return self.update(detections, frame_idx)
