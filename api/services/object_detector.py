"""
Object detector wrapper for the API perception service.
Uses YOLOv8 to detect all interacting objects across standard classes.
"""
import numpy as np
from typing import List, Dict, Any
from models.perception.detection.yolo_detector import YOLODetector

class ObjectDetector:
    def __init__(self):
        # Initialize YOLO for multi-object detection (excluding person, which is class 0)
        self.detector = YOLODetector(model_size="m")
        self.detector.initialize()
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all objects in frame and map to dictionary format."""
        detections = self.detector.detect(frame)
        
        objects = []
        for det in detections:
            # We skip 'person' here if it's already handled by the pose estimator,
            # or keep it as a base for tracking.
            # In the user's snippet, 'persons' and 'objects' are separate.
            if det.class_name == 'person':
                continue
                
            objects.append({
                "bbox": det.bbox.tolist(), # [x1, y1, x2, y2]
                "type": det.class_name,
                "confidence": float(det.confidence),
                "class_id": det.class_id
            })
            
        return objects
