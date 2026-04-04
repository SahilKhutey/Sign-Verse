"""
Multi-object and person tracker wrapper for perception service.
Uses ByteTrack to manage IDs across frames.
"""
from typing import List, Dict, Any
from models.perception.tracking.bytetrack import ByteTracker, Track

class Tracker:
    def __init__(self):
        # We use two independent trackers for objects and persons to simplify ID management
        self.person_tracker = ByteTracker(max_lost=30)
        self.object_tracker = ByteTracker(max_lost=30)
        self.frame_idx = 0
        
    def track_persons(self, poses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign persistent IDs to detected persons."""
        self.frame_idx += 1
        
        # ByteTracker.update expects a list of detection-like objects with .bbox and .confidence
        # We'll wrap the poses in a simple object for this purpose
        class PersonDetection:
            def __init__(self, bbox, confidence):
                self.bbox = bbox
                self.confidence = confidence

        person_dets = [PersonDetection(p['bbox'], p['confidence']) for p in poses]
        tracks = self.person_tracker.update(person_dets, self.frame_idx)
        
        # Map tracked IDs back to pose structures
        # (Assuming the association is simple for this mock-like service)
        tracked_poses = []
        for track in tracks:
            p = next((pose for pose in poses if self._bbox_match(pose['bbox'], track.bbox)), None)
            if p:
                p['id'] = track.track_id
                tracked_poses.append(p)
                
        return tracked_poses
    
    def track_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign persistent IDs to detected objects."""
        class ObjectDetection:
            def __init__(self, bbox, confidence):
                self.bbox = bbox
                self.confidence = confidence

        obj_dets = [ObjectDetection(o['bbox'], o['confidence']) for o in objects]
        tracks = self.object_tracker.update(obj_dets, self.frame_idx)
        
        tracked_objects = []
        for track in tracks:
            o = next((obj for obj in objects if self._bbox_match(obj['bbox'], track.bbox)), None)
            if o:
                o['id'] = track.track_id
                tracked_objects.append(o)
                
        return tracked_objects

    def _bbox_match(self, bbox1, bbox2, iou_thresh=0.5):
        """Simple IoU check for matching."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        iou = inter / (area1 + area2 - inter + 1e-6)
        return iou >= iou_thresh
