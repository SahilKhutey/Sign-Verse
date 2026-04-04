"""
Multi-class detector for humans and objects using YOLOv8/v9.
Extends the SignVerse perception layer for human-object interaction understanding.
"""
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from ultralytics import YOLO

@dataclass
class EntityDetection:
    """Detection result for any entity (person or object)."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    entity_type: str  # "person" or "object"
    features: Optional[np.ndarray] = None

class MultiClassDetector:
    """YOLO-based detector for multiple object classes including humans."""
    
    # COCO class names for common objects
    COCO_CLASS_NAMES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
        35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
        39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
        44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
        49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
        59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
        64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
        69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
        74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
        79: "toothbrush"
    }
    
    # Custom class mappings for specific applications
    CUSTOM_CLASS_GROUPS = {
        "fitness": ["dumbbell", "barbell", "bench", "treadmill", "exercise_ball"],
        "kitchen": ["knife", "spoon", "fork", "cup", "bowl", "bottle"],
        "office": ["laptop", "mouse", "keyboard", "cell phone", "book"],
        "tools": ["hammer", "screwdriver", "wrench", "scissors"]
    }
    
    def __init__(self, model_size: str = "m", confidence_threshold: float = 0.5,
                 classes: Optional[List[int]] = None, custom_classes: Optional[List[str]] = None):
        """
        Initialize multi-class detector.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detection
            classes: List of COCO class IDs to detect
            custom_classes: List of custom class groups
        """
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        self.classes = classes or list(range(80))  # All COCO classes by default
        self.custom_classes = custom_classes or []
        self.initialized = False
        
        # Expand custom class groups
        self._expand_custom_classes()
        
        self.model = None
        logger.info(f"Initializing MultiClassDetector (YOLOv8-{self.model_size}) with {len(self.classes)} classes")
    
    def _expand_custom_classes(self):
        """Expand custom class groups to individual class IDs."""
        expanded_classes = set(self.classes)
        
        for group_name in self.custom_classes:
            if group_name in self.CUSTOM_CLASS_GROUPS:
                for class_name in self.CUSTOM_CLASS_GROUPS[group_name]:
                    # Find class ID by name
                    found = False
                    for class_id, name in self.COCO_CLASS_NAMES.items():
                        if name == class_name:
                            expanded_classes.add(class_id)
                            found = True
                            break
                    if not found:
                        logger.warning(f"Custom class '{class_name}' not found in COCO mapping.")
        
        self.classes = list(expanded_classes)
    
    def initialize(self):
        """Initialize YOLO model and weights."""
        try:
            self.model = YOLO(f"yolov8{self.model_size}.pt")
            self.initialized = True
            logger.success(f"Successfully initialized YOLOv8-{self.model_size} multi-class detector")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[EntityDetection]:
        """Detect all entities (humans and objects) in the current frame."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Execute inference
            results = self.model(frame, verbose=False, classes=self.classes)
            detections = []
            
            for result in results:
                if result.boxes is None:
                    continue
                
                for box in result.boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    
                    if confidence >= self.confidence_threshold:
                        bbox = box.xyxy[0].cpu().numpy()
                        class_name = self.COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")
                        
                        # Categorize entity
                        entity_type = "person" if class_id == 0 else "object"
                        
                        detection = EntityDetection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            entity_type=entity_type
                        )
                        detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} entities: {[d.class_name for d in detections]}")
            return detections
            
        except Exception as e:
            logger.error(f"Multi-class detection failed: {e}")
            return []
    
    def detect_specific_classes(self, frame: np.ndarray, class_names: List[str]) -> List[EntityDetection]:
        """Detect only a subset of specific classes defined by name."""
        class_ids = []
        for class_name in class_names:
            found = False
            for class_id, name in self.COCO_CLASS_NAMES.items():
                if name == class_name:
                    class_ids.append(class_id)
                    found = True
                    break
            if not found:
                logger.warning(f"Requested class '{class_name}' for specific detection not in COCO.")
        
        if not class_ids:
            return []
        
        # Temporarily restrict classes for this call
        original_classes = self.classes
        self.classes = class_ids
        
        detections = self.detect(frame)
        
        # Restore original broad class list
        self.classes = original_classes
        
        return detections
    
    def extract_features(self, frame: np.ndarray, detection: EntityDetection) -> np.ndarray:
        """Extract appearance-based features for object tracking and ReID."""
        # Extract ROI from detection bbox
        x1, y1, x2, y2 = map(int, detection.bbox)
        # Ensure bbox within frame boundaries
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.array([])
        
        features = []
        
        # 1. Color Histogram Features (8 bins per channel)
        for channel in range(3):
            hist = cv2.calcHist([roi], [channel], None, [8], [0, 256])
            # Normalize histogram
            cv2.normalize(hist, hist)
            features.extend(hist.flatten())
        
        # 2. Textural/Gradient Features (Spatial derivatives)
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Reduce resolution for texture stats if ROI is large
            if gray.shape[0] > 100 or gray.shape[1] > 100:
                gray = cv2.resize(gray, (64, 64))
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Use basic statistical descriptors of gradients
            features.extend([
                np.mean(sobelx), np.mean(sobely), 
                np.std(sobelx), np.std(sobely)
            ])
        
        return np.array(features)
