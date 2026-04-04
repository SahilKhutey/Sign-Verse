"""
SignVerse Perception System Test Script
Complete validation of multi-person tracking, object detection, pose estimation,
depth awareness, and interaction detection in real-time.

Usage Examples:
    # Webcam testing
    python scripts/system_test.py --source 0

    # Video file testing
    python scripts/system_test.py --source input_video.mp4 --output output_video.avi

    # Headless mode (for processing only)
    python scripts/system_test.py --source input_video.mp4 --headless

    # Specific camera resolution
    python scripts/system_test.py --source 0 --resolution 1920 1080
"""
import cv2
import numpy as np
import torch
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import argparse

# -------------------------------
# CONFIGURATION
# -------------------------------
@dataclass
class SystemConfig:
    """System configuration parameters."""
    # Model settings
    yolo_model_size: str = "n"  # n, s, m, l, x
    yolo_confidence: float = 0.5
    yolo_classes: List[int] = None  # None for all classes
    
    # Depth estimation
    depth_model: str = "MiDaS_small"
    depth_visualization: bool = True
    
    # Pose estimation
    pose_model_complexity: int = 1
    pose_min_confidence: float = 0.5
    
    # Tracking
    max_track_age: int = 30
    min_detection_confidence: float = 0.4
    
    # Interaction detection
    interaction_threshold: float = 0.3
    hand_proximity_threshold: int = 30  # pixels
    
    # Visualization
    show_fps: bool = True
    show_depth: bool = True
    show_interactions: bool = True
    show_tracking_ids: bool = True
    
    # Performance
    target_fps: int = 30
    resolution: Tuple[int, int] = (1280, 720)

# -------------------------------
# MODEL LOADING
# -------------------------------
class ModelLoader:
    """Loads and manages all AI models."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all required models."""
        print("Loading YOLO detection model...")
        self.models['yolo'] = YOLO(f"yolov8{self.config.yolo_model_size}.pt")
        
        print("Loading MiDaS depth estimation model...")
        self.models['depth'] = torch.hub.load("intel-isl/MiDaS", self.config.depth_model)
        self.models['depth'].eval()
        self.models['depth_transform'] = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        
        print("Loading MediaPipe Holistic model...")
        self.models['holistic'] = mp.solutions.holistic.Holistic(
            model_complexity=self.config.pose_model_complexity,
            min_detection_confidence=self.config.pose_min_confidence,
            min_tracking_confidence=self.config.pose_min_confidence
        )
        
        print("All models loaded successfully!")
    
    def get_model(self, model_name: str):
        """Get a specific model."""
        return self.models.get(model_name)

# -------------------------------
# TRACKING SYSTEM
# -------------------------------
@dataclass
class TrackedEntity:
    """Tracked entity with history."""
    entity_id: str
    entity_type: str  # "person", "object"
    class_name: str
    bbox: np.ndarray
    confidence: float
    history: deque
    last_seen: int
    
    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=30)

class MultiObjectTracker:
    """Tracks both people and objects with temporal consistency."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.next_id = {"person": 1, "object": 1}
        self.tracks: Dict[str, TrackedEntity] = {}
        self.frame_count = 0
    
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[TrackedEntity]:
        """Update tracks with new detections."""
        self.frame_count += 1
        updated_tracks = []
        
        for detection in detections:
            entity_type = detection['entity_type']
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Find existing track or create new one
            track_id = self._find_existing_track(bbox, entity_type)
            
            if track_id is None:
                track_id = f"{entity_type[0].upper()}{self.next_id[entity_type]}"
                self.next_id[entity_type] += 1
                
                new_track = TrackedEntity(
                    entity_id=track_id,
                    entity_type=entity_type,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    history=deque([bbox], maxlen=30),
                    last_seen=self.frame_count
                )
                self.tracks[track_id] = new_track
                updated_tracks.append(new_track)
            else:
                # Update existing track
                track = self.tracks[track_id]
                track.bbox = bbox
                track.confidence = confidence
                track.history.append(bbox)
                track.last_seen = self.frame_count
                updated_tracks.append(track)
        
        # Remove old tracks
        self._cleanup_old_tracks()
        
        return updated_tracks
    
    def _find_existing_track(self, bbox: np.ndarray, entity_type: str) -> Optional[str]:
        """Find existing track using IoU matching."""
        best_iou = 0.3  # Minimum IoU threshold
        best_track_id = None
        
        for track_id, track in self.tracks.items():
            if track.entity_type != entity_type:
                continue
            
            if self.frame_count - track.last_seen > self.config.max_track_age:
                continue
            
            iou = self._calculate_iou(bbox, track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id
        
        return best_track_id
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if self.frame_count - track.last_seen > self.config.max_track_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]

# -------------------------------
# DEPTH ESTIMATION
# -------------------------------
class DepthEstimator:
    """Handles depth estimation and spatial reasoning."""
    
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map from single image."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        return depth.astype(np.uint8)
    
    def estimate_entity_depth(self, depth_map: np.ndarray, bbox: np.ndarray) -> float:
        """Estimate depth for a specific entity."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = depth_map[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0
        
        return float(np.median(roi))

# -------------------------------
# INTERACTION DETECTION
# -------------------------------
class InteractionDetector:
    """Detects human-object interactions."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def detect_interactions(self, persons: List[TrackedEntity], 
                          objects: List[TrackedEntity],
                          depth_map: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Detect interactions between persons and objects."""
        interactions = []
        
        for person in persons:
            if person.entity_type != "person":
                continue
            
            for obj in objects:
                if obj.entity_type != "object":
                    continue
                
                # Check spatial proximity
                if self._check_proximity(person, obj, depth_map):
                    interaction_type = self._determine_interaction_type(person, obj)
                    confidence = self._calculate_confidence(person, obj)
                    
                    if interaction_type and confidence > self.config.interaction_threshold:
                        interactions.append({
                            "person_id": person.entity_id,
                            "object_id": obj.entity_id,
                            "type": interaction_type,
                            "confidence": confidence,
                            "frame": person.last_seen
                        })
        
        return interactions
    
    def _check_proximity(self, person: TrackedEntity, obj: TrackedEntity, 
                        depth_map: Optional[np.ndarray]) -> bool:
        """Check if person and object are close enough for interaction."""
        # Calculate 2D distance between centers
        person_center = np.array([(person.bbox[0] + person.bbox[2]) / 2,
                                (person.bbox[1] + person.bbox[3]) / 2])
        obj_center = np.array([(obj.bbox[0] + obj.bbox[2]) / 2,
                             (obj.bbox[1] + obj.bbox[3]) / 2])
        
        distance_2d = np.linalg.norm(person_center - obj_center)
        
        # Use depth information if available
        if depth_map is not None:
            person_depth = self._estimate_entity_depth(depth_map, person.bbox)
            obj_depth = self._estimate_entity_depth(depth_map, obj.bbox)
            depth_diff = abs(person_depth - obj_depth)
            
            # Combined distance metric
            distance = np.sqrt(distance_2d**2 + depth_diff**2)
        else:
            distance = distance_2d
        
        return distance < self.config.hand_proximity_threshold
    
    def _estimate_entity_depth(self, depth_map: np.ndarray, bbox: np.ndarray) -> float:
        """Estimate depth for an entity."""
        x1, y1, x2, y2 = map(int, bbox)
        roi = depth_map[y1:y2, x1:x2]
        return float(np.median(roi)) if roi.size > 0 else 0.0
    
    def _determine_interaction_type(self, person: TrackedEntity, obj: TrackedEntity) -> str:
        """Determine the type of interaction."""
        # Simple heuristic-based interaction typing
        # In production, this would use more sophisticated rules or ML
        
        object_type = obj.class_name.lower()
        
        if object_type in ["cup", "bottle", "glass"]:
            return "holding"
        elif object_type in ["phone", "cell phone"]:
            return "using"
        elif object_type in ["book", "laptop"]:
            return "interacting_with"
        else:
            return "near"
    
    def _calculate_confidence(self, person: TrackedEntity, obj: TrackedEntity) -> float:
        """Calculate interaction confidence."""
        # Simple confidence calculation based on proximity and duration
        proximity = 1.0 - (self._calculate_distance(person.bbox, obj.bbox) / 
                          self.config.hand_proximity_threshold)
        
        # Temporal consistency (longer presence = higher confidence)
        temporal = min(len(person.history) / 10.0, 1.0)
        
        return (proximity * 0.7) + (temporal * 0.3)
    
    def _calculate_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate distance between two bounding boxes."""
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        return np.linalg.norm(center1 - center2)

# -------------------------------
# VISUALIZATION
# -------------------------------
class VisualizationEngine:
    """Handles visualization of all system outputs."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.colors = {
            "person": (0, 255, 0),      # Green
            "object": (255, 0, 0),      # Blue
            "interaction": (0, 0, 255), # Red
            "text": (255, 255, 255)     # White
        }
        self.fps_history = deque(maxlen=30)
    
    def draw_detections(self, frame: np.ndarray, entities: List[TrackedEntity]) -> np.ndarray:
        """Draw detected entities on frame."""
        for entity in entities:
            color = self.colors.get(entity.entity_type, (255, 255, 255))
            x1, y1, x2, y2 = map(int, entity.bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{entity.entity_id} {entity.class_name}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_interactions(self, frame: np.ndarray, interactions: List[Dict[str, Any]]) -> np.ndarray:
        """Draw interaction information on frame."""
        for i, interaction in enumerate(interactions):
            label = f"{interaction['person_id']} -> {interaction['object_id']}: {interaction['type']}"
            cv2.putText(frame, label, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['interaction'], 2)
        
        return frame
    
    def draw_depth_map(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Draw depth map visualization."""
        depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        return np.hstack([frame, depth_colormap])
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter on frame."""
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history) if self.fps_history else fps
        
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        return frame

# -------------------------------
# MAIN SYSTEM
# -------------------------------
class PerceptionSystem:
    """Main perception system that integrates all components."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.tracker = MultiObjectTracker(config)
        self.depth_estimator = DepthEstimator(
            self.model_loader.models['depth'],
            self.model_loader.models['depth_transform']
        )
        self.interaction_detector = InteractionDetector(config)
        self.visualizer = VisualizationEngine(config)
        
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame through the complete system."""
        self.frame_count += 1
        results = {
            "frame_number": self.frame_count,
            "timestamp": time.time() - self.start_time,
            "entities": [],
            "interactions": [],
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Object detection
            detections = self._detect_objects(frame)
            
            # Step 2: Tracking
            tracked_entities = self.tracker.update(detections, frame)
            results["entities"] = tracked_entities
            
            # Step 3: Pose estimation
            pose_results = self._estimate_poses(frame, tracked_entities)
            
            # Step 4: Depth estimation
            depth_map = self.depth_estimator.estimate_depth(frame)
            results["depth_map"] = depth_map
            
            # Step 5: Interaction detection
            persons = [e for e in tracked_entities if e.entity_type == "person"]
            objects = [e for e in tracked_entities if e.entity_type == "object"]
            
            interactions = self.interaction_detector.detect_interactions(
                persons, objects, depth_map
            )
            results["interactions"] = interactions
            
            # Step 6: Visualization
            visualization = self._create_visualization(frame, tracked_entities, 
                                                     interactions, depth_map)
            results["visualization"] = visualization
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            results["error"] = str(e)
        
        finally:
            results["processing_time"] = time.time() - start_time
        
        return results
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame using YOLO."""
        results = self.model_loader.models['yolo'](frame, verbose=False)[0]
        detections = []
        
        for r in results.boxes.data:
            x1, y1, x2, y2, conf, cls = r
            if conf < self.config.yolo_confidence:
                continue
            
            class_id = int(cls)
            class_name = results.names[class_id]
            entity_type = "person" if class_id == 0 else "object"
            
            detections.append({
                "entity_type": entity_type,
                "class_name": class_name,
                "bbox": np.array([x1, y1, x2, y2.cpu()]),
                "confidence": float(conf)
            })
        
        return detections
    
    def _estimate_poses(self, frame: np.ndarray, entities: List[TrackedEntity]) -> List[Dict[str, Any]]:
        """Estimate poses for detected persons."""
        pose_results = []
        
        for entity in entities:
            if entity.entity_type != "person":
                continue
            
            # Extract ROI for pose estimation
            x1, y1, x2, y2 = map(int, entity.bbox)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Process with MediaPipe
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.model_loader.models['holistic'].process(rgb_roi)
            
            pose_results.append({
                "person_id": entity.entity_id,
                "pose_landmarks": results.pose_landmarks,
                "left_hand_landmarks": results.left_hand_landmarks,
                "right_hand_landmarks": results.right_hand_landmarks,
                "face_landmarks": results.face_landmarks
            })
        
        return pose_results
    
    def _create_visualization(self, frame: np.ndarray, entities: List[TrackedEntity],
                             interactions: List[Dict[str, Any]], depth_map: np.ndarray) -> np.ndarray:
        """Create comprehensive visualization."""
        # Draw detections and tracking
        visualization = self.visualizer.draw_detections(frame.copy(), entities)
        
        # Draw interactions
        if self.config.show_interactions:
            visualization = self.visualizer.draw_interactions(visualization, interactions)
        
        # Draw depth map
        if self.config.show_depth:
            visualization = self.visualizer.draw_depth_map(visualization, depth_map)
        
        # Draw FPS
        if self.config.show_fps:
            elapsed = time.time() - self.start_time
            current_fps = self.frame_count / elapsed if elapsed > 0 else 0
            visualization = self.visualizer.draw_fps(visualization, current_fps)
        
        return visualization

# -------------------------------
# MAIN EXECUTION
# -------------------------------
def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="SignVerse Perception System Test")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--output", type=str, help="Output video path (optional)")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    args = parser.parse_args()
    
    # Initialize system
    config = SystemConfig()
    system = PerceptionSystem(config)
    
    # Open video source
    try:
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {args.source}")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
        
        # Output video writer
        out = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(args.output, fourcc, 30.0, config.resolution)
        
        print("Starting perception system...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = system.process_frame(frame)
            
            if not args.headless:
                # Show visualization
                cv2.imshow("SignVerse Perception System", results["visualization"])
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"screenshot_{timestamp}.jpg", results["visualization"])
                    print(f"Screenshot saved: screenshot_{timestamp}.jpg")
            
            # Write to output video
            if out is not None:
                out.write(results["visualization"])
            
            # Print progress
            if system.frame_count % 30 == 0:
                print(f"Processed {system.frame_count} frames | "
                      f"Entities: {len(results['entities'])} | "
                      f"Interactions: {len(results['interactions'])} | "
                      f"FPS: {1/results['processing_time']:.1f}")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
