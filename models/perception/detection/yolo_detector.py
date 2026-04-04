"""
YOLOv8 detector implementation.
"""
import cv2
import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from loguru import logger

from ..detector_base import BaseDetector, Detection

class YOLODetector(BaseDetector):
    """YOLOv8-based object detector."""
    
    def __init__(self, 
                 model_size: str = "m",
                 confidence_threshold: float = 0.5,
                 classes: Optional[List[int]] = None):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for detection
            classes: List of class IDs to detect (None for all)
        """
        super().__init__(confidence_threshold)
        self.model_size = model_size
        self.classes = classes
        self.model = None
    
    def initialize(self):
        """Initialize YOLOv8 model."""
        try:
            self.model = YOLO(f"yolov8{self.model_size}.pt")
            self.initialized = True
            logger.success(f"Initialized YOLOv8-{self.model_size} detector")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Preprocess frame
            processed_frame = self.preprocess(frame)
            
            # Run detection
            results = self.model(processed_frame, verbose=False, classes=self.classes)
            
            # Extract detections
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        class_name = self.model.names[class_id]
                        
                        if confidence >= self.confidence_threshold:
                            bbox = box.xyxy[0].cpu().numpy()
                            detection = Detection(
                                bbox=bbox,
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name
                            )
                            detections.append(detection)
            
            # Postprocess
            detections = self.postprocess(detections)
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batch detection for multiple frames."""
        if not self.initialized:
            self.initialize()
        
        all_detections = []
        for frame in frames:
            detections = self.detect(frame)
            all_detections.append(detections)
        
        return all_detections
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO."""
        # YOLO handles preprocessing internally, but we can add custom preprocessing here
        return frame
    
    def postprocess(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression and other postprocessing."""
        # YOLO includes NMS internally, but we can add additional filtering
        return super().postprocess(detections)
