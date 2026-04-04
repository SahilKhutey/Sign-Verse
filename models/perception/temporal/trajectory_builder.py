"""
Trajectory building for temporal consistency and smoothing.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from loguru import logger

class TrajectoryBuilder:
    """Builds and maintains temporal trajectories for pose landmarks."""
    
    def __init__(self, window_size: int = 10, smoothing: bool = True):
        """
        Initialize trajectory builder.
        
        Args:
            window_size: Size of trajectory window
            smoothing: Whether to apply temporal smoothing
        """
        self.window_size = window_size
        self.smoothing = smoothing
        self.trajectories = defaultdict(list)
        self.current_frame = 0
    
    def update(self, 
              track_id: int,
              pose_data: Dict[str, Any],
              frame_idx: int) -> Dict[str, Any]:
        """
        Update trajectories with new pose data.
        
        Args:
            track_id: Tracking ID
            pose_data: Current pose data
            frame_idx: Frame index
            
        Returns:
            Smoothed pose data with temporal consistency
        """
        self.current_frame = frame_idx
        
        # Store current pose in trajectory
        for landmark_type, landmarks in pose_data.items():
            if isinstance(landmarks, np.ndarray):
                key = f"{track_id}_{landmark_type}"
                self.trajectories[key].append({
                    'frame': frame_idx,
                    'landmarks': landmarks,
                    'timestamp': frame_idx / 30.0  # Assuming 30 FPS
                })
                
                # Maintain window size
                if len(self.trajectories[key]) > self.window_size:
                    self.trajectories[key].pop(0)
        
        # Apply temporal smoothing if enabled
        if self.smoothing:
            return self._apply_temporal_smoothing(track_id, pose_data)
        else:
            return pose_data
    
    def _apply_temporal_smoothing(self, track_id: int, current_pose: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal smoothing to pose data."""
        smoothed_pose = {}
        
        for landmark_type, current_landmarks in current_pose.items():
            key = f"{track_id}_{landmark_type}"
            
            if key in self.trajectories and len(self.trajectories[key]) > 1:
                # Get historical landmarks
                history = self.trajectories[key]
                landmarks_history = [item['landmarks'] for item in history]
                
                # Ensure all landmarks have same shape
                if all(lm.shape == current_landmarks.shape for lm in landmarks_history):
                    # Stack landmarks for processing
                    landmarks_stack = np.stack(landmarks_history + [current_landmarks])
                    
                    # Apply moving average smoothing
                    smoothed = np.mean(landmarks_stack, axis=0)
                    smoothed_pose[landmark_type] = smoothed
                else:
                    smoothed_pose[landmark_type] = current_landmarks
            else:
                smoothed_pose[landmark_type] = current_landmarks
        
        return smoothed_pose
    
    def predict(self, track_id: int, steps: int = 1) -> Optional[Dict[str, Any]]:
        """Predict future pose based on trajectory."""
        predictions = {}
        
        for landmark_type in set(key.split('_')[1] for key in self.trajectories.keys() if key.startswith(f"{track_id}_")):
            key = f"{track_id}_{landmark_type}"
            
            if key in self.trajectories and len(self.trajectories[key]) >= 2:
                history = self.trajectories[key]
                
                # Simple linear prediction based on recent velocity
                recent_frames = history[-2:]
                if len(recent_frames) == 2:
                    lm1 = recent_frames[0]['landmarks']
                    lm2 = recent_frames[1]['landmarks']
                    dt = recent_frames[1]['timestamp'] - recent_frames[0]['timestamp']
                    
                    if dt > 0:
                        velocity = (lm2 - lm1) / dt
                        predicted = lm2 + velocity * steps * (1/30.0)  # Assume 30 FPS
                        predictions[landmark_type] = predicted
        
        return predictions if predictions else None
    
    def get_trajectory(self, track_id: int, landmark_type: str) -> List[Any]:
        """Get full trajectory for specific landmark type."""
        key = f"{track_id}_{landmark_type}"
        return self.trajectories.get(key, [])
    
    def clear_trajectory(self, track_id: int, landmark_type: Optional[str] = None):
        """Clear trajectory for track or specific landmark type."""
        if landmark_type:
            key = f"{track_id}_{landmark_type}"
            if key in self.trajectories:
                del self.trajectories[key]
        else:
            # Clear all trajectories for this track
            keys_to_remove = [key for key in self.trajectories.keys() if key.startswith(f"{track_id}_")]
            for key in keys_to_remove:
                del self.trajectories[key]
