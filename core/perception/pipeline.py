"""
Main perception pipeline that orchestrates all components.
"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import time
import os

from models.perception.detection.yolo_detector import YOLODetector
from models.perception.tracking.bytetrack import ByteTracker
from models.perception.pose.holistic import HolisticPoseEstimator
from models.perception.features.velocity import calculate_joint_velocities
from models.perception.features.acceleration import calculate_joint_accelerations
from models.perception.features.gestures import detect_hand_gestures
from models.perception.temporal.smoothing import TemporalSmoother
from models.perception.temporal.interpolation import interpolate_full_pose
from ..storage.router import StorageTierRouter
from ..data_models.skeleton import SkeletonFrame, BodyJoints, HandJoints, FaceLandmarks, HeadPose, Vector3D
from ..data_models.trajectory import PersonTrajectory, TemporalTracker

@dataclass
class PipelineConfig:
    detection_confidence: float = 0.6
    tracking_max_lost: int = 30
    model_complexity: int = 2
    pose_confidence: float = 0.5
    normalization: bool = True
    feature_extraction: bool = True
    storage_config: Dict[str, Any] = field(default_factory=lambda: {
        "file_system": {"base_path": "./data"},
        "database": {"database_url": "sqlite:///./data/signverse.db"},
        "hpc": {"base_path": "./data/hpc", "format": "parquet"}
    })

class PerceptionPipeline:
    """Main perception pipeline that coordinates all components."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the complete perception pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.detector = YOLODetector(confidence_threshold=self.config.detection_confidence)
        self.tracker = ByteTracker(max_lost=self.config.tracking_max_lost)
        self.pose_estimator = HolisticPoseEstimator(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.pose_confidence,
            min_tracking_confidence=self.config.pose_confidence
        )
        self.smoother = TemporalSmoother()
        self.storage = StorageTierRouter(self.config.storage_config)
        self.storage.initialize()
        self.temporal_tracker = TemporalTracker(max_gap_frames=self.config.tracking_max_lost)
        
        # State for temporal features
        self.previous_poses = {} # track_id -> pose_dict
        self.previous_velocities = {} # track_id -> velocity_dict
        
        self.frame_count = 0
        self.processing_times = []
        
        logger.info("Initialized Perception Pipeline")
    
    def process_frame(self, frame: np.ndarray) -> Dict[int, Any]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary of results per track ID
        """
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # Step 1: Person detection
            detections = self.detector.detect(frame)
            
            # Step 2: Tracking
            tracks = self.tracker.update(detections, self.frame_count)
            
            results = {}
            
            # Step 3: Pose estimation for each tracked person
            for track in tracks:
                try:
                    # ROI Processing via the estimator
                    pose_result = self.pose_estimator.estimate(frame, track.bbox)
                    
                    # Step 4: Temporal Processing (Smoothing & Interpolation)
                    if self.config.normalization:
                        # 1. Fill missing data
                        landmarks = interpolate_full_pose(pose_result.landmarks, pose_result.confidence)
                        
                        # 2. Smooth
                        landmarks = self.smoother.smooth(track.track_id, landmarks)
                    else:
                        landmarks = pose_result.landmarks
                    
                    # Step 5: Feature extraction
                    features = {}
                    if self.config.feature_extraction:
                        dt = 1/30.0 # Assuming 30 FPS
                        
                        # Calculate motion features
                        if track.track_id in self.previous_poses:
                            vels = calculate_joint_velocities(landmarks, self.previous_poses[track.track_id], dt)
                            features['velocities'] = vels
                            
                            if track.track_id in self.previous_velocities:
                                accels = calculate_joint_accelerations(vels, self.previous_velocities[track.track_id], dt)
                                features['accelerations'] = accels
                            self.previous_velocities[track.track_id] = vels
                        
                        self.previous_poses[track.track_id] = landmarks
                        
                        # Detect discrete states
                        features['gestures'] = detect_hand_gestures(
                            landmarks.get('left_hand'), 
                            landmarks.get('right_hand')
                        )
                        features['expressions'] = classify_facial_expressions(landmarks.get('face'))
                    
                    # Step 6: Create Unified Skeleton Frame
                    skeleton_frame = SkeletonFrame(
                        frame_id=self.frame_count,
                        timestamp=time.time() - start_time, # Simple relative timestamp
                        person_id=f"P{track.track_id}",
                        overall_confidence=pose_result.confidence,
                        tracking_quality=1.0, # Placeholder
                        processing_time=time.time() - start_time
                    )
                    # (In a real scenario, we would map landmarks properly here)
                    # Step 7: Storage & Temporal Tracking
                    self.storage.store_frame(skeleton_frame)
                    self.temporal_tracker.update(skeleton_frame)
                    
                    # Store results
                    results[track.track_id] = {
                        'track': track,
                        'landmarks': landmarks,
                        'features': features,
                        'bbox': track.bbox
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing track {track.track_id}: {e}")
                    continue
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Log performance
            if self.frame_count % 30 == 0:
                avg_time = np.mean(self.processing_times[-30:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                logger.info(f"Processing: {fps:.1f} FPS, {len(results)} persons tracked")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {}
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict:
        """
        Process complete video file.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        all_results = {}
        processed_frames = 0
        
        logger.info(f"Processing video: {video_path} ({total_frames} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and processed_frames >= max_frames):
                break
            
            results = self.process_frame(frame)
            all_results[processed_frames] = results
            processed_frames += 1
            
            if processed_frames % 100 == 0:
                logger.info(f"Processed {processed_frames}/{total_frames} frames")
        
        cap.release()
        
        # Step 8: Store all trajectories
        all_trajectories = self.temporal_tracker.get_all_trajectories()
        for person_id, trajectory in all_trajectories.items():
            logger.info(f"Archiving trajectory for {person_id} ({len(trajectory.trajectory)} frames)")
            self.storage.store_trajectory(trajectory)
            
        # Calculate statistics
        stats = self._calculate_statistics(all_results)
        logger.info(f"Video processing completed: {stats}")
        
        return {
            'results': all_results,
            'statistics': stats,
            'total_frames': processed_frames,
            'trajectories': all_trajectories
        }
    
    def _pose_result_to_dict(self, pose_result) -> Dict[str, Any]:
        """Convert PoseResult to dictionary."""
        return {
            'body_landmarks': pose_result.body_landmarks,
            'left_hand_landmarks': pose_result.left_hand_landmarks,
            'right_hand_landmarks': pose_result.right_hand_landmarks,
            'face_landmarks': pose_result.face_landmarks,
            'pose_world_landmarks': pose_result.pose_world_landmarks,
            'segmentation_mask': pose_result.segmentation_mask
        }
    
    def _calculate_statistics(self, all_results: Dict) -> Dict[str, Any]:
        """Calculate processing statistics."""
        total_persons = 0
        total_frames_with_persons = 0
        confidence_scores = []
        
        for frame_results in all_results.values():
            if frame_results:
                total_frames_with_persons += 1
                total_persons += len(frame_results)
                
                for track_data in frame_results.values():
                    if (track_data.get('normalized_pose') and 
                        track_data['normalized_pose'].confidence_scores):
                        for scores in track_data['normalized_pose'].confidence_scores.values():
                            if isinstance(scores, np.ndarray):
                                confidence_scores.extend(scores.flatten())
                            elif isinstance(scores, list):
                                confidence_scores.extend(scores)
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        avg_persons_per_frame = total_persons / len(all_results) if all_results else 0
        
        return {
            'total_frames': len(all_results),
            'frames_with_persons': total_frames_with_persons,
            'total_persons_detected': total_persons,
            'avg_persons_per_frame': avg_persons_per_frame,
            'avg_confidence': avg_confidence,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0
        }
    
    def close(self):
        """Release all resources."""
        self.pose_estimator.close()
        logger.info("Perception pipeline closed")
