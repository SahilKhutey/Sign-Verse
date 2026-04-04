"""
Skeleton fusion for combining multiple pose estimation results.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

class SkeletonFusion:
    """Fuses multiple pose estimation results into a unified skeleton."""
    
    def __init__(self, fusion_method: str = "weighted_average"):
        """
        Initialize skeleton fusion.
        
        Args:
            fusion_method: Fusion method ('weighted_average', 'kalman', 'rts_smoother')
        """
        self.fusion_method = fusion_method
        self.pose_history = {}
    
    def fuse_poses(self, 
                  current_pose: Dict[str, Any],
                  previous_pose: Optional[Dict[str, Any]] = None,
                  confidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fuse current pose with previous poses.
        
        Args:
            current_pose: Current pose estimation
            previous_pose: Previous pose estimation(s)
            confidence: Confidence scores for current pose
            
        Returns:
            Fused pose result
        """
        if self.fusion_method == "weighted_average":
            return self._weighted_average_fusion(current_pose, previous_pose, confidence)
        elif self.fusion_method == "kalman":
            return self._kalman_fusion(current_pose, previous_pose, confidence)
        else:
            return current_pose
    
    def _weighted_average_fusion(self, 
                               current_pose: Dict[str, Any],
                               previous_pose: Optional[Dict[str, Any]],
                               confidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted average fusion based on confidence."""
        if previous_pose is None or confidence is None:
            return current_pose
        
        fused_pose = {}
        
        for landmark_type, current_landmarks in current_pose.items():
            if landmark_type in previous_pose and landmark_type in confidence:
                prev_landmarks = previous_pose[landmark_type]
                conf = confidence[landmark_type]
                
                # Ensure shapes match
                if (isinstance(current_landmarks, np.ndarray) and 
                    isinstance(prev_landmarks, np.ndarray) and
                    current_landmarks.shape == prev_landmarks.shape):
                    
                    # Expand confidence to match landmarks shape if needed
                    if isinstance(conf, np.ndarray) and conf.ndim == 1:
                        conf = conf[:, np.newaxis]
                    
                    # Weighted average
                    fused = (current_landmarks * conf + prev_landmarks * (1 - conf))
                    fused_pose[landmark_type] = fused
                else:
                    fused_pose[landmark_type] = current_landmarks
            else:
                fused_pose[landmark_type] = current_landmarks
        
        return fused_pose
    
    def _kalman_fusion(self, 
                      current_pose: Dict[str, Any],
                      previous_pose: Optional[Dict[str, Any]],
                      confidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Kalman filter fusion for smoother results."""
        # Simplified Kalman implementation
        # In production, you'd use a proper Kalman filter implementation
        fused_pose = {}
        
        for landmark_type, current_landmarks in current_pose.items():
            if (landmark_type in previous_pose and 
                isinstance(current_landmarks, np.ndarray) and
                isinstance(previous_pose[landmark_type], np.ndarray) and
                current_landmarks.shape == previous_pose[landmark_type].shape):
                
                # Simple smoothing based on confidence
                if confidence and landmark_type in confidence:
                    conf = confidence[landmark_type]
                    if isinstance(conf, np.ndarray) and conf.ndim == 1:
                        conf = conf[:, np.newaxis]
                    
                    # Kalman-like update (simplified)
                    fused = previous_pose[landmark_type] + conf * (current_landmarks - previous_pose[landmark_type])
                    fused_pose[landmark_type] = fused
                else:
                    # Moving average
                    alpha = 0.8  # Smoothing factor
                    fused = alpha * previous_pose[landmark_type] + (1 - alpha) * current_landmarks
                    fused_pose[landmark_type] = fused
            else:
                fused_pose[landmark_type] = current_landmarks
        
        return fused_pose
    
    def update_history(self, track_id: int, pose: Dict[str, Any], frame_idx: int):
        """Update pose history for a track."""
        if track_id not in self.pose_history:
            self.pose_history[track_id] = []
        
        self.pose_history[track_id].append({
            'frame': frame_idx,
            'pose': pose,
            'timestamp': frame_idx / 30.0  # Assuming 30 FPS
        })
        
        # Keep only recent history
        if len(self.pose_history[track_id]) > 30:  # Keep 1 second at 30 FPS
            self.pose_history[track_id].pop(0)
    
    def get_historical_pose(self, track_id: int, frames_back: int = 1) -> Optional[Dict[str, Any]]:
        """Get historical pose for a track."""
        if (track_id in self.pose_history and 
            len(self.pose_history[track_id]) >= frames_back):
            return self.pose_history[track_id][-frames_back]['pose']
        return None
