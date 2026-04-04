"""
Spatial Interaction Mapper for Human-Object Interaction (HOI) understanding.
Calculates relationships between tracked persons and detected objects.
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..data_models.skeleton import SkeletonFrame, Vector3D, JointType
from .multi_detector import EntityDetection

class InteractionType(str, Enum):
    """Types of detected interactions."""
    PROXIMITY = "proximity"   # Person is near the object
    TOUCH = "touch"           # Person's hand is touching/overlapping the object
    HOLD = "hold"             # Person is holding/carrying the object (Temporal)
    USE = "use"               # Person is actively using the object
    OBSERVE = "observe"       # Person is looking at the object (Eye contact)

@dataclass
class InteractionState:
    """Represents a specific interaction between a person and an entity."""
    person_id: str
    entity_class: str
    interaction_type: InteractionType
    confidence: float
    distance: float  # Estimated 3D or 2D distance
    metadata: Dict[str, Any] = field(default_factory=dict)

class SpatialInteractionMapper:
    """Maps spatial relationships between skeletons and detected entities."""
    
    def __init__(self, proximity_threshold: float = 0.2, touch_threshold: float = 0.05):
        """
        Initialize the interaction mapper.
        
        Args:
            proximity_threshold: Distance threshold for 'proximity' interaction
            touch_threshold: Distance threshold for 'touch' interaction
        """
        self.proximity_threshold = proximity_threshold
        self.touch_threshold = touch_threshold
        logger.info("Initialized SpatialInteractionMapper")

    def map_interactions(self, skeleton: SkeletonFrame, 
                        detections: List[EntityDetection]) -> List[InteractionState]:
        """
        Map interactions for a single person against all visible detections.
        
        Args:
            skeleton: Standardized skeleton frame for one person
            detections: List of all entity detections in the frame
            
        Returns:
            List of detected interaction states
        """
        interactions = []
        
        # 1. Extract reference points from skeleton
        # For HOI, we primarily care about hands and head orientation
        hands = {
            "left": skeleton.left_hand.wrist if skeleton.left_hand else None,
            "right": skeleton.right_hand.wrist if skeleton.right_hand else None
        }
        
        head = skeleton.head_pose
        
        for detection in detections:
            # Skip self-detection if person is in detection list
            if detection.entity_type == "person" and detection.class_name == "person":
                # In most cases class_id 0 is the skeleton we're processing
                # We can add person-person interaction logic here later
                continue
                
            # Calculate object centroid (normalized 0-1)
            obj_x = (detection.bbox[0] + detection.bbox[2]) / 2.0
            obj_y = (detection.bbox[1] + detection.bbox[3]) / 2.0
            # Note: detections are in pixel space or normalized? 
            # MultiClassDetector returns xyxy from YOLO which is usually pixel space or relative to frame.
            # Skeleton uses normalized coordinates. We need to unify these.
            
            # Assuming detection.bbox is normalized [0, 1] for this calculation
            # If not, we'd need the frame resolution.
            obj_pos = np.array([obj_x, obj_y])
            
            # Check for proximity and touch
            for hand_side, hand_wrist in hands.items():
                if hand_wrist is None: continue
                
                hand_pos = np.array([hand_wrist.x, hand_wrist.y])
                dist = np.linalg.norm(hand_pos - obj_pos)
                
                # Check for Touch (Hand-Object overlap or very close proximity)
                if dist < self.touch_threshold:
                    interactions.append(InteractionState(
                        person_id=skeleton.person_id,
                        entity_class=detection.class_name,
                        interaction_type=InteractionType.TOUCH,
                        confidence=min(1.0, (1.0 - dist/self.touch_threshold)),
                        distance=dist,
                        metadata={"hand": hand_side}
                    ))
                # Check for Proximity
                elif dist < self.proximity_threshold:
                    interactions.append(InteractionState(
                        person_id=skeleton.person_id,
                        entity_class=detection.class_name,
                        interaction_type=InteractionType.PROXIMITY,
                        confidence=min(1.0, (1.0 - dist/self.proximity_threshold)),
                        distance=dist,
                        metadata={"hand": hand_side}
                    ))
            
            # Check for Observation (Gaze/Head Pose)
            if head:
                # Basic gaze logic: check if object centroid is in the direction of head yaw/pitch
                # This requires mapping yaw/pitch to a 2D screen vector from the head centroid
                pass
                
        return interactions

    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]) + (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1]) - intersection
        
        return intersection / union if union > 0 else 0.0

    def batch_map(self, skeletons: List[SkeletonFrame], 
                 detections: List[EntityDetection]) -> Dict[str, List[InteractionState]]:
        """Map interactions for multiple persons in a frame."""
        batch_results = {}
        for skeleton in skeletons:
            batch_results[skeleton.person_id] = self.map_interactions(skeleton, detections)
        return batch_results
