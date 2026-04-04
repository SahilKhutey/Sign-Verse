"""
Enhanced Perception Service for SignVerse backend.
Provides object detection, pose estimation, and interaction mapping.
"""
import cv2
import numpy as np
from typing import Dict, List, Any
import torch

from .pose_estimator import PoseEstimator
from .object_detector import ObjectDetector
from .tracker import Tracker

class PerceptionService:
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.object_detector = ObjectDetector()
        self.tracker = Tracker()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the perception pipeline"""
        results = {
            "persons": [],
            "objects": [],
            "interactions": [],
            "timestamp": 0,  # Would be set based on video timing
            "frame_size": frame.shape[:2]
        }
        
        # Detect objects
        objects = self.object_detector.detect(frame)
        results["objects"] = objects
        
        # Detect poses
        poses = self.pose_estimator.estimate(frame)
        # (Pass poses list through tracker)
        results["persons"] = self.tracker.track_persons(poses)
        
        # Track objects
        results["objects"] = self.tracker.track_objects(objects)
        
        # Detect interactions between tracked personas and objects
        interactions = self.detect_interactions(results["persons"], results["objects"])
        results["interactions"] = interactions
        
        return results
    
    def detect_interactions(self, persons: List[Dict], objects: List[Dict]) -> List[Dict]:
        """Detect interactions between persons and objects with refined semantics for robotics."""
        interactions = []
        
        for person in persons:
            person_bbox = person.get("bbox") or self._get_person_bbox(person)
            
            for obj in objects:
                obj_bbox = obj["bbox"]
                
                # Check for intersection between person and object bounding boxes
                overlap = self._bbox_intersection(person_bbox, obj_bbox)
                if overlap > 0.25:  # Slightly lower threshold for robot-like interactions
                    interaction = {
                        "type": "interaction",
                        "person_id": person.get("id", "unknown"),
                        "object_id": obj.get("id", "unknown"),
                        "object_type": obj.get("type", "unknown"),
                        "confidence": min(person.get("confidence", 0), obj.get("confidence", 0)),
                        "bbox": obj_bbox,
                        "overlap": overlap
                    }
                    
                    # Refine interaction type based on pose and object type
                    obj_type = obj.get("type")
                    if obj_type in ["cup", "bottle", "glass"]:
                        interaction["type"] = "drinking"
                    elif obj_type in ["phone", "cellphone"]:
                        interaction["type"] = "using_phone"
                    elif obj_type in ["book", "laptop", "tablet"]:
                        interaction["type"] = "reading"
                    elif obj_type in ["chair", "couch", "bed"]:
                        interaction["type"] = "sitting" if obj_type == "chair" else "resting"
                    elif obj_type in ["scissors", "knife", "fork"]:
                        interaction["type"] = "manipulating_tool"
                    elif obj_type in ["keyboard", "mouse"]:
                        interaction["type"] = "typing" if obj_type == "keyboard" else "interacting_input"
                    elif obj_type in ["sports ball", "frisbee"]:
                        interaction["type"] = "playing"
                        
                    interactions.append(interaction)
        
        return interactions
    
    def _get_person_bbox(self, person: Dict) -> List[float]:
        """Get bounding box from pose keypoints"""
        keypoints = person.get("keypoints", [])
        if not keypoints:
            return [0, 0, 0, 0]
        
        xs = [kp["x"] for kp in keypoints if kp["score"] > 0.2]
        ys = [kp["y"] for kp in keypoints if kp["score"] > 0.2]
        
        if not xs or not ys:
            return [0, 0, 0, 0]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add some padding
        padding = 20
        return [
            max(0, x_min - padding),
            max(0, y_min - padding),
            x_max + padding,
            y_max + padding
        ]
    
    def _bbox_intersection(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate intersection over union between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / min(area1, area2) if min(area1, area2) > 0 else 0

# Global service instance
perception_service = PerceptionService()

def process_frame(frame: np.ndarray) -> Dict[str, Any]:
    """Process a frame using the perception service"""
    return perception_service.process_frame(frame)
