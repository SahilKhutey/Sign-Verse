"""
ByteTrack implementation for multi-object tracking.
"""
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from loguru import logger

from ..tracker_base import BaseTracker, Track

class ByteTracker(BaseTracker):
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box."""
    
    def __init__(self, 
                 max_lost: int = 30,
                 track_thresh: float = 0.6,
                 high_thresh: float = 0.7,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):
        super().__init__(max_lost)
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.track_buffer = frame_rate * 2  # 2 seconds buffer
        
    def initialize(self):
        """Initialize ByteTracker."""
        self.initialized = True
        logger.success("Initialized ByteTracker")
    
    def update(self, detections: List[Any], frame_idx: int) -> List[Track]:
        """Update tracks with new detections."""
        if not self.initialized:
            self.initialize()
        
        # Separate detections by confidence
        high_conf_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_conf_dets = [d for d in detections if self.track_thresh <= d.confidence < self.high_thresh]
        
        # Predict current tracks
        predicted_tracks = self.predict()
        
        # First association: high confidence detections
        matches, unmatched_tracks, unmatched_dets = self._associate(
            predicted_tracks, high_conf_dets, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = predicted_tracks[track_idx]
            detection = high_conf_dets[det_idx]
            
            self._update_track(track, detection, frame_idx)
            high_conf_dets[det_idx] = None  # Mark as matched
        
        # Second association: low confidence detections with unmatched tracks
        remaining_dets = [d for d in high_conf_dets if d is not None] + low_conf_dets
        matches2, unmatched_tracks2, unmatched_dets2 = self._associate(
            unmatched_tracks, remaining_dets, self.match_thresh * 0.5
        )
        
        for track_idx, det_idx in matches2:
            track = unmatched_tracks[track_idx]
            detection = remaining_dets[det_idx]
            
            self._update_track(track, detection, frame_idx)
            remaining_dets[det_idx] = None
        
        # Create new tracks for remaining detections
        for detection in remaining_dets:
            if detection is not None and detection.confidence >= self.track_thresh:
                self._create_new_track(detection, frame_idx)
        
        # Mark lost tracks
        self._mark_lost_tracks(unmatched_tracks2, frame_idx)
        
        return self.get_active_tracks()
    
    def _associate(self, tracks: List[Track], detections: List[Any], iou_thresh: float):
        """Associate tracks with detections using IoU."""
        if not tracks or not detections:
            return [], tracks, detections
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection.bbox)
        
        # Hungarian algorithm for assignment
        matched_indices = self._hungarian_assignment(iou_matrix, iou_thresh)
        
        # Extract matches and unmatched
        matches = []
        unmatched_tracks = []
        unmatched_dets = list(range(len(detections)))
        
        for i, j in matched_indices:
            if iou_matrix[i, j] >= iou_thresh:
                matches.append((i, j))
                if j in unmatched_dets:
                    unmatched_dets.remove(j)
        
        # Find unmatched tracks
        for i in range(len(tracks)):
            if i not in [idx for idx, _ in matches]:
                unmatched_tracks.append(tracks[i])
        
        # Find unmatched detections
        unmatched_detections = [detections[j] for j in unmatched_dets]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray, threshold: float):
        """Hungarian algorithm for assignment."""
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Maximize IoU
            matches = []
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] >= threshold:
                    matches.append((i, j))
            return matches
        except ImportError:
            # Fallback to greedy assignment
            matches = []
            assigned_rows = set()
            assigned_cols = set()
            
            # Sort by IoU descending
            indices = np.dstack(np.unravel_index(np.argsort(-cost_matrix.ravel()), cost_matrix.shape))[0]
            
            for i, j in indices:
                if i not in assigned_rows and j not in assigned_cols and cost_matrix[i, j] >= threshold:
                    matches.append((int(i), int(j)))
                    assigned_rows.add(i)
                    assigned_cols.add(j)
            
            return matches
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)
    
    def _update_track(self, track: Track, detection: Any, frame_idx: int):
        """Update existing track with new detection."""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.features = getattr(detection, 'features', track.features)
        track.frame_count += 1
        track.lost_count = 0
        
        # Update history
        if track.history is None:
            track.history = []
        track.history.append({
            'frame': frame_idx,
            'bbox': detection.bbox,
            'confidence': detection.confidence
        })
    
    def _create_new_track(self, detection: Any, frame_idx: int):
        """Create new track from detection."""
        track_id = self.next_id
        self.next_id += 1
        
        track = Track(
            track_id=track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            features=getattr(detection, 'features', None),
            frame_count=1,
            lost_count=0,
            history=[{
                'frame': frame_idx,
                'bbox': detection.bbox,
                'confidence': detection.confidence
            }]
        )
        
        self.tracks[track_id] = track
    
    def _mark_lost_tracks(self, tracks: List[Track], frame_idx: int):
        """Mark tracks as lost."""
        for track in tracks:
            track.lost_count += 1
            if track.lost_count > self.max_lost:
                if track.track_id in self.tracks:
                    del self.tracks[track.track_id]
    
    def predict(self) -> List[Track]:
        """Predict next state of tracks using simple linear prediction."""
        predicted_tracks = []
        for track in self.tracks.values():
            if track.lost_count == 0 and track.history and len(track.history) >= 2:
                # Simple linear prediction based on history
                last_frame = track.history[-1]
                prev_frame = track.history[-2]
                
                # Calculate velocity
                dt = last_frame['frame'] - prev_frame['frame']
                if dt > 0:
                    velocity = (last_frame['bbox'] - prev_frame['bbox']) / dt
                    predicted_bbox = last_frame['bbox'] + velocity
                    
                    # Create predicted track
                    predicted_track = Track(
                        track_id=track.track_id,
                        bbox=predicted_bbox,
                        confidence=track.confidence * 0.9,  # Reduce confidence for prediction
                        features=track.features,
                        frame_count=track.frame_count,
                        lost_count=track.lost_count,
                        history=track.history
                    )
                    predicted_tracks.append(predicted_track)
                else:
                    predicted_tracks.append(track)
            else:
                predicted_tracks.append(track)
        
        return predicted_tracks
