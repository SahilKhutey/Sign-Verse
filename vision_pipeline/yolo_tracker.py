"""
YOLO Person Tracker

Isolates the primary signer via a high-speed YOLO bounding box.
Stabilizes the vision pipeline for downstream keypoint extractors.
"""
import cv2
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics.YOLO not found. To install: `pip install ultralytics`")

class YoloTracker:
    def __init__(self, model_name="yolov8n.pt", confidence=0.4):
        self.mock_mode = (YOLO is None)
        self.conf = confidence
        if not self.mock_mode:
            # We explicitly load tracking for Person (Class 0 in COCO)
            self.model = YOLO(model_name)
        
    def get_signer_bbox(self, frame):
        """
        Runs YOLO to find the largest Person (Class 0) bounding box.
        Returns None if no person is detected.
        """
        if self.mock_mode:
            return None
            
        results = self.model.predict(source=frame, classes=[0], conf=self.conf, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
            
        boxes = results[0].boxes
        # Get the largest box by Area
        max_area = 0
        best_box = None
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (int(x1), int(y1), int(x2), int(y2))
                
        return best_box

    def extract_crop(self, frame, bbox, padding=0.1):
        """
        Extracts a stabilized cropped frame padded efficiently.
        Returns the crop and its absolute frame coordinates.
        """
        if bbox is None:
            return frame, (0, 0, frame.shape[1], frame.shape[0])
            
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add padding (10% around the bounding box to capture full arm extensions)
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        nx1 = max(0, x1 - pad_w)
        ny1 = max(0, y1 - pad_h)
        nx2 = min(w, x2 + pad_w)
        ny2 = min(h, y2 + pad_h)
        
        crop = frame[ny1:ny2, nx1:nx2]
        return crop, (nx1, ny1, nx2, ny2)
