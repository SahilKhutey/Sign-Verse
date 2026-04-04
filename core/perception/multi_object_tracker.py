"""
Multi-object tracker for both humans and objects in the SignVerse ecosystem.
Maintains separate ID spaces and temporal history for various entity types.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger

from .multi_detector import EntityDetection

@dataclass
class TrackedEntity:
    """Tracked entity with temporal history and state."""
    track_id: str  # "P1", "O1", "O2"
    entity_type: str  # "person", "object"
    class_name: str  # "person", "dumbbell", "car"
    bbox: np.ndarray
    confidence: float
    features: Any
    frame_count: int
    lost_count: int
    history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []

class MultiObjectTracker:
    """Tracks both humans and objects with separate ID spaces and temporal persistence."""
    
    def __init__(self, max_lost: int = 30, iou_threshold: float = 0.3):
        """
        Initialize multi-object tracker.
        
        Args:
            max_lost: Maximum frames to keep lost tracks before purging
            iou_threshold: Minimum IoU for valid track-detection association
        """
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        
        # Separate auto-incrementing counters for different entity categories
        self.next_ids = {"person": 1, "object": 1}
        self.tracks: Dict[str, TrackedEntity] = {}  # Active track_id -> TrackedEntity
        self.lost_tracks: Dict[str, TrackedEntity] = {} # Temporal buffer for lost tracks
        
        logger.info("Initialized MultiObjectTracker with separate ID spaces")
    
    def update(self, detections: List[EntityDetection], frame_idx: int) -> List[TrackedEntity]:
        """
        Update tracks with new detections from the multi-class detector.
        
        Args:
            detections: List of entity detections for the current frame
            frame_idx: Current frame index for history tracking
            
        Returns: active_tracks
            List of all active tracked entities (humans and objects)
        """
        # Segment detections by base type to avoid cross-type ID assignment
        person_detections = [d for d in detections if d.entity_type == "person"]
        object_detections = [d for d in detections if d.entity_type == "object"]
        
        # Sequentially update tracking states for each entity domain
        person_tracks = self._update_entities(person_detections, "person", frame_idx)
        object_tracks = self._update_entities(object_detections, "object", frame_idx)
        
        return person_tracks + object_tracks
    
    def _update_entities(self, detections: List[EntityDetection], entity_type: str, 
                        frame_idx: int) -> List[TrackedEntity]:
        """Core update logic for a specific entity category."""
        # Collate existing candidates (active + recently lost)
        existing_tracks = [t for t in self.tracks.values() if t.entity_type == entity_type]
        existing_tracks += [t for t in self.lost_tracks.values() if t.entity_type == entity_type]
        
        # Global Association via Hungarian/Linear Sum Assignment
        matches, unmatched_tracks, unmatched_dets = self._associate(
            existing_tracks, detections, self.iou_threshold
        )
        
        updated_tracks = []
        updated_ids = set()
        
        # Propagate temporal state for matched pairings
        for track_idx, det_idx in matches:
            track = existing_tracks[track_idx]
            detection = detections[det_idx]
            
            self._update_track(track, detection, frame_idx)
            updated_tracks.append(track)
            updated_ids.add(track.track_id)
            
            # Nullify processed detection in the input list
            detections[det_idx] = None
        
        # Spawn new tracks for previously unseen entities
        for detection in detections:
            if detection is not None:
                new_track = self._create_new_track(detection, entity_type, frame_idx)
                updated_tracks.append(new_track)
                updated_ids.add(new_track.track_id)
        
        # Handle tracks that were not successfully matched in this frame
        self._mark_lost_tracks(existing_tracks, updated_ids, frame_idx)
        
        return updated_tracks
    
    def _associate(self, tracks: List[TrackedEntity], detections: List[EntityDetection],
                  iou_threshold: float):
        """Associate existing tracks with new detections using spatial overlap."""
        if not tracks or not detections:
            return [], tracks, list(range(len(detections)))
        
        # Construct cost matrix based on Intersection over Union (IoU)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = self._calculate_iou(track.bbox, detection.bbox)
        
        # Optimize assignment
        matched_indices = self._hungarian_assignment(cost_matrix, iou_threshold)
        
        matches = []
        unmatched_dets = list(range(len(detections)))
        
        for i, j in matched_indices:
            matches.append((i, j))
            if j in unmatched_dets:
                unmatched_dets.remove(j)
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in [idx for idx, _ in matches]]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _update_track(self, track: TrackedEntity, detection: EntityDetection, frame_idx: int):
        """Update tracked entity state with new visual evidence."""
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.features = detection.features
        track.frame_count += 1
        track.lost_count = 0
        
        # Append to temporal trajectory
        track.history.append({
            "frame": frame_idx,
            "bbox": detection.bbox.tolist(),
            "confidence": detection.confidence,
            "class_name": detection.class_name
        })
        
        # Revive from lost buffer if necessary
        if track.track_id in self.lost_tracks:
            self.tracks[track.track_id] = track
            del self.lost_tracks[track.track_id]
    
    def _create_new_track(self, detection: EntityDetection, entity_type: str, frame_idx: int) -> TrackedEntity:
        """Initialize a new track ID space."""
        track_id = f"{entity_type[0].upper()}{self.next_ids[entity_type]}"  # P1, O1, etc.
        self.next_ids[entity_type] += 1
        
        track = TrackedEntity(
            track_id=track_id,
            entity_type=entity_type,
            class_name=detection.class_name,
            bbox=detection.bbox,
            confidence=detection.confidence,
            features=detection.features,
            frame_count=1,
            lost_count=0,
            history=[{
                "frame": frame_idx,
                "bbox": detection.bbox.tolist(),
                "confidence": detection.confidence,
                "class_name": detection.class_name
            }]
        )
        
        self.tracks[track_id] = track
        return track
    
    def _mark_lost_tracks(self, tracks: List[TrackedEntity], updated_ids: set, frame_idx: int):
        """Update lost status for unmatched tracks."""
        for track in tracks:
            if track.track_id not in updated_ids:
                track.lost_count += 1
                
                if track.lost_count > self.max_lost:
                    # Final purge from system
                    self.tracks.pop(track.track_id, None)
                    self.lost_tracks.pop(track.track_id, None)
                else:
                    # Stash in lost buffer for potential re-assignment
                    if track.track_id in self.tracks:
                        self.lost_tracks[track.track_id] = track
                        del self.tracks[track.track_id]
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Standard IoU calculation for rectangular overlap."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)
    
    def _hungarian_assignment(self, cost_matrix: np.ndarray, threshold: float):
        """Solves optimal assignment problem using Hungarian algorithm."""
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Maximize IoU (negative cost)
            
            matches = []
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] >= threshold:
                    matches.append((i, j))
            return matches
        except ImportError:
            # Greedy fallback if scipy is unavailable
            matches = []
            assigned_rows = set()
            assigned_cols = set()
            
            # Sort indices by overlap confidence descending
            indices = np.dstack(np.unravel_index(np.argsort(-cost_matrix.ravel()), cost_matrix.shape))[0]
            
            for i, j in indices:
                if i not in assigned_rows and j not in assigned_cols and cost_matrix[i, j] >= threshold:
                    matches.append((int(i), int(j)))
                    assigned_rows.add(i)
                    assigned_cols.add(j)
            
            return matches
    
    def get_entity_trajectory(self, track_id: str) -> List[Dict[str, Any]]:
        """Retrieve temporal position history for a given entity."""
        if track_id in self.tracks:
            return self.tracks[track_id].history
        elif track_id in self.lost_tracks:
            return self.lost_tracks[track_id].history
        return []
    
    def get_active_entities(self) -> List[TrackedEntity]:
        """List all entities currently being actively tracked."""
        return list(self.tracks.values())
    
    def get_entities_by_type(self, entity_type: str) -> List[TrackedEntity]:
        """Filter active tracks by type (person or object)."""
        return [t for t in self.tracks.values() if t.entity_type == entity_type]
