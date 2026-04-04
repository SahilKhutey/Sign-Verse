"""
Temporal smoothing filters for pose landmark sequences.
"""
import numpy as np
from typing import Dict, List, Any, Optional

class TemporalSmoother:
    """Applies moving average smoothing to landmark sequences."""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the smoother.
        
        Args:
            window_size: Number of frames to use for moving average
        """
        self.window_size = window_size
        self.history = {}  # track_id -> List[Dict[landmark_type, array]]
        
    def smooth(self, track_id: int, current_landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Smooth current landmarks using historical window.
        
        Args:
            track_id: Tracking ID for temporal consistency
            current_landmarks: Current frame's landmark arrays
            
        Returns:
            Smoothed landmark arrays
        """
        if track_id not in self.history:
            self.history[track_id] = []
            
        # Add current state to history
        self.history[track_id].append({k: v.copy() for k, v in current_landmarks.items()})
        
        # Maintain window size
        if len(self.history[track_id]) > self.window_size:
            self.history[track_id].pop(0)
            
        # Perform smoothing
        if len(self.history[track_id]) <= 1:
            return current_landmarks
            
        smoothed_landmarks = {}
        for landmark_type in current_landmarks.keys():
            # Only smooth if this type is present in all frames of the current window
            if all(landmark_type in frame for frame in self.history[track_id]):
                frames = [frame[landmark_type] for frame in self.history[track_id]]
                smoothed_landmarks[landmark_type] = np.mean(frames, axis=0)
            else:
                smoothed_landmarks[landmark_type] = current_landmarks[landmark_type]
                
        return smoothed_landmarks

    def reset(self, track_id: Optional[int] = None):
        """Reset history for a specific track or all tracks."""
        if track_id is not None:
            if track_id in self.history:
                del self.history[track_id]
        else:
            self.history = {}
