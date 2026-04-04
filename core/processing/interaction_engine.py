"""
Human-Object Interaction (HOI) Detection Engine for SignVerse OS.
Derives behavioral intent from spatial, temporal, and pose-contextual cues.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger
from scipy.spatial.distance import cdist

@dataclass
class Interaction:
    """Represents a detected behavioral interaction between a person and an entity."""
    person_id: str
    object_id: str
    interaction_type: str  # "holding", "touching", "using", "looking_at"
    confidence: float
    start_frame: int
    end_frame: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class InteractionEngine:
    """Tracks and classifies human-object interactions across temporal windows."""
    
    # Heuristic-based behavioral thresholds
    INTERACTION_TYPES = {
        "holding": {
            "max_distance": 0.15,  # Normalized 3D or 2D distance
            "min_duration": 5,     # Required frames for temporal consistency
            "required_motion": True
        },
        "touching": {
            "max_distance": 0.20,
            "min_duration": 3,
            "required_motion": False
        },
        "using": {
            "max_distance": 0.25,
            "min_duration": 10,
            "required_motion": True
        },
        "looking_at": {
            "max_distance": 0.4,
            "min_duration": 5,
            "required_motion": False
        }
    }
    
    def __init__(self, spatial_threshold: float = 0.25, temporal_window: int = 15):
        """
        Initialize the interaction engine.
        
        Args:
            spatial_threshold: Universal maximum distance for any interaction
            temporal_window: Buffer size for smoothing behavioral states
        """
        self.spatial_threshold = spatial_threshold
        self.temporal_window = temporal_window
        self.interaction_history: Dict[str, List[Interaction]] = {}
        self.active_interactions: Dict[str, Interaction] = {}
        
        logger.info("Initialized SignVerse InteractionEngine (HOI Analysis Tier)")
    
    def detect_interactions(self, frame_data: Dict[str, Any]) -> List[Interaction]:
        """
        Synchronize with frame data to extract behavioral interactions.
        
        Args:
            frame_data: Aggregated data structure including persons, objects, and depth maps
            
        Returns: current_frame_interactions
            List of interactions active in this frame
        """
        interactions = []
        frame_id = frame_data.get("frame_id", 0)
        persons = frame_data.get("persons", [])
        objects = frame_data.get("objects", [])
        depth_map = frame_data.get("depth_map")
        intrinsic_matrix = frame_data.get("intrinsic_matrix")
        
        for person in persons:
            for obj in objects:
                # Phase 1: Spatial Proximity (Gating)
                dist_normalized = self._calculate_distance(person, obj)
                
                # Refine with Depth if available
                if depth_map is not None and intrinsic_matrix is not None:
                    dist_3d = self._calculate_depth_aware_distance(
                        person, obj, depth_map, intrinsic_matrix
                    )
                    # We use the minimum between normalized 2D and 3D distance for robustness
                    dist_normalized = min(dist_normalized, dist_3d)
                
                if dist_normalized < self.spatial_threshold:
                    # Phase 2: Behavioral Context Analysis
                    motion_corr = self._check_motion_correlation(person, obj)
                    pose_ctx = self._check_pose_context(person, obj)
                    
                    # Phase 3: Classification
                    i_type = self._determine_interaction_type(person, obj, motion_corr, pose_ctx)
                    
                    if i_type:
                        confidence = self._calculate_confidence(person, obj, motion_corr, pose_ctx)
                        
                        interaction = Interaction(
                            person_id=person["id"],
                            object_id=obj["id"],
                            interaction_type=i_type,
                            confidence=confidence,
                            start_frame=frame_id,
                            metadata={
                                "dist": float(dist_normalized),
                                "motion_corr": motion_corr,
                                "context": pose_ctx
                            }
                        )
                        
                        interactions.append(interaction)
                        self._update_interaction_history(interaction)
        
        return interactions
    
    def _calculate_distance(self, person: Dict[str, Any], obj: Dict[str, Any]) -> float:
        """Calculate normalized center-to-center distance."""
        p_bbox = person["bbox"]
        o_bbox = obj["bbox"]
        
        p_center = np.array([(p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2])
        o_center = np.array([(o_bbox[0] + o_bbox[2]) / 2, (o_bbox[1] + o_bbox[3]) / 2])
        
        # Norm based on a standard resolution to maintain threshold consistency
        res = person.get("res", (1280, 720))
        dist_px = np.linalg.norm(p_center - o_center)
        return dist_px / np.linalg.norm(res)
    
    def _calculate_depth_aware_distance(self, person: Dict[str, Any], obj: Dict[str, Any],
                                      depth_map: np.ndarray, intrinsic_matrix: np.ndarray) -> float:
        """Calculate metric-estimated 3D distance between entity centroids."""
        try:
            p_pos = self._estimate_3d_position(person["bbox"], depth_map, intrinsic_matrix)
            o_pos = self._estimate_3d_position(obj["bbox"], depth_map, intrinsic_matrix)
            
            dist_3d = np.linalg.norm(p_pos - o_pos)
            # Normalize for engine thresholds (Mapping ~2 meters to 1.0)
            return dist_3d / 2.0
        except Exception as e:
            return 1.0 # Return large value on failure

    def _estimate_3d_position(self, bbox: np.ndarray, depth_map: np.ndarray,
                            intrinsic_matrix: np.ndarray) -> np.ndarray:
        """Backproject bounding box centroid to 3D."""
        u = (bbox[0] + bbox[2]) / 2.0
        v = (bbox[1] + bbox[3]) / 2.0
        
        depth = depth_map[int(v), int(u)] if 0 <= int(v) < depth_map.shape[0] and 0 <= int(u) < depth_map.shape[1] else 0.5
        
        x = (u - intrinsic_matrix[0, 2]) * depth / intrinsic_matrix[0, 0]
        y = (v - intrinsic_matrix[1, 2]) * depth / intrinsic_matrix[1, 1]
        return np.array([x, y, depth])
    
    def _check_motion_correlation(self, person: Dict[str, Any], obj: Dict[str, Any]) -> float:
        """Measure motion trajectory similarity (Dummy logic for now)."""
        return 0.75 if person.get("is_moving") and obj.get("is_moving") else 0.2
    
    def _check_pose_context(self, person: Dict[str, Any], obj: Dict[str, Any]) -> Dict[str, float]:
        """Examine skeletal landmarks for interaction specific cues."""
        ctx = {}
        
        if "hands" in person:
            ctx["hand_proximity"] = InteractionUtils.detect_hand_object_interaction(
                person["hands"], obj["bbox"]
            )
            
        if "gaze" in person:
            ctx["gaze_alignment"] = InteractionUtils.estimate_gaze_direction(
                person["gaze"], obj["bbox"]
            )
            
        return ctx
    
    def _determine_interaction_type(self, person: Dict[str, Any], obj: Dict[str, Any],
                                  motion_corr: float, pose_ctx: Dict[str, float]) -> Optional[str]:
        """Categorize the interaction based on spatial and skeletal telemetry."""
        hand_prox = pose_ctx.get("hand_proximity", 0)
        gaze_align = pose_ctx.get("gaze_alignment", 0)
        
        # Holding detection (Contact + Motion mapping)
        if hand_prox > 0.8 and motion_corr > 0.6:
            return "holding"
        
        # Touching detection (Contact only)
        if hand_prox > 0.7:
            return "touching"
        
        # Observing detection (Gaze mapping)
        if gaze_align > 0.7:
            return "looking_at"
            
        return None
    
    def _calculate_confidence(self, person: Dict[str, Any], obj: Dict[str, Any],
                            motion_corr: float, pose_ctx: Dict[str, float]) -> float:
        """Aggregated behavioral confidence score."""
        scores = [
            0.4 * pose_ctx.get("hand_proximity", 0.5),
            0.3 * motion_corr,
            0.3 * pose_ctx.get("gaze_alignment", 0.5)
        ]
        return min(sum(scores), 1.0)
    
    def _update_interaction_history(self, interaction: Interaction):
        """Append to temporal buffer for behavioral persistence analysis."""
        key = f"{interaction.person_id}_{interaction.object_id}"
        if key not in self.interaction_history:
            self.interaction_history[key] = []
        
        self.interaction_history[key].append(interaction)
        if len(self.interaction_history[key]) > self.temporal_window:
            self.interaction_history[key].pop(0)

class InteractionUtils:
    """Low-level algorithmic utilities for behavioral checks."""
    
    @staticmethod
    def detect_hand_object_interaction(hand_positions: List[np.ndarray], 
                                     object_bbox: np.ndarray,
                                     depth_map: Optional[np.ndarray] = None) -> float:
        """Measure proximity between hand landmarks and target object."""
        if not hand_positions: return 0.0
        
        obj_center = np.array([(object_bbox[0] + object_bbox[2]) / 2, (object_bbox[1] + object_bbox[3]) / 2])
        min_dist = float('inf')
        
        for h_pos in hand_positions:
            dist_2d = np.linalg.norm(h_pos[:2] - obj_center)
            if depth_map is not None:
                # Add depth Z component to distance
                z_diff = abs(depth_map[int(h_pos[1]), int(h_pos[0])] - depth_map[int(obj_center[1]), int(obj_center[0])]) if 0 <= int(h_pos[1]) < depth_map.shape[0] else 0.5
                dist = np.sqrt(dist_2d**2 + (z_diff * 100)**2) # Weighted Z
            else:
                dist = dist_2d
            min_dist = min(min_dist, dist)
            
        # Transform distance to 0-1 confidence (Threshold at ~100px relative)
        return max(0, 1.0 - (min_dist / 100.0))
    
    @staticmethod
    def estimate_gaze_direction(gaze_vector: np.ndarray, 
                              object_bbox: np.ndarray) -> float:
        """Calculate alignment between head/eye gaze and object centroid."""
        # gaze_vector is assumed normalized (3D)
        obj_center = np.array([(object_bbox[0]+object_bbox[2])/2, (object_bbox[1]+object_bbox[3])/2, 0.5])
        # Vector to object (Relative 3D approximation)
        to_obj = obj_center - np.array([0.5, 0.5, 0.0]) # Relative from center
        to_obj /= np.linalg.norm(to_obj)
        
        return max(0, float(np.dot(gaze_vector, to_obj)))
