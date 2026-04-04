"""
Pose estimator wrapper for the API perception service.
Uses HolisticPoseEstimator for detailed skeleton and landmark detection.
"""
import numpy as np
from typing import List, Dict, Any
from models.perception.pose.holistic import HolisticPoseEstimator
from models.perception.detection.yolo_detector import YOLODetector

class PoseEstimator:
    def __init__(self):
        # We use YOLO to find people first for multi-person support
        self.detector = YOLODetector(model_size="n", classes=[0]) # class 0 is 'person'
        self.holistic = HolisticPoseEstimator(model_complexity=1)
        self.detector.initialize()
        self.holistic.initialize()
        
    def estimate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect all persons and estimate their holistic poses."""
        # Step 1: Detect persons
        detections = self.detector.detect(frame)
        
        poses = []
        for det in detections:
            # Step 2: Estimate pose for each person's ROI
            pose_result = self.holistic.estimate(frame, det.bbox)
            
            # Map landmarks to the requested keypoint format
            keypoints = []
            if 'body' in pose_result.landmarks:
                for i, lm in enumerate(pose_result.landmarks['body']):
                    keypoints.append({
                        "id": i,
                        "x": float(lm[0] * frame.shape[1]), # Convert normalized to pixel coords
                        "y": float(lm[1] * frame.shape[0]),
                        "score": float(pose_result.confidence['body'][i]) if 'body' in pose_result.confidence else 0.8
                    })
            
            poses.append({
                "keypoints": keypoints,
                "confidence": float(det.confidence),
                "bbox": det.bbox.tolist()
            })
            
        return poses
