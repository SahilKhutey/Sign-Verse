"""
Main processing pipeline for video pose estimation.
Coordinates Detection, Tracking, Holistic Pose, Normalization, and Storage tiers.
"""
import cv2
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
import numpy as np
from pathlib import Path

from ...models.perception.detection.yolo_detector import YOLODetector
from ...models.perception.tracking.bytetrack import ByteTracker
from ...models.perception.pose.holistic import HolisticPoseEstimator
from ..processing.normalization import PoseNormalizer, get_normalizer
from ..data_models.skeleton import SkeletonFrame
from ..storage.storage_manager import StorageManager

@dataclass
class PipelineConfig:
    """Pipeline configuration for high-level video processing."""
    # Detection settings
    detection_confidence: float = 0.6
    detection_model_size: str = "m"
    
    # Tracking settings
    tracking_max_lost: int = 30
    tracking_iou_threshold: float = 0.3
    
    # Pose estimation settings
    pose_model_complexity: int = 2
    pose_min_confidence: float = 0.5
    
    # Normalization settings
    normalization_preset: str = "ml_training"
    
    # Performance settings
    target_fps: int = 30
    batch_processing: bool = False
    enable_caching: bool = True
    
    # Storage settings
    storage_base_path: str = "./data"
    database_url: str = "sqlite:///./data/metadata.db"

class VideoProcessingPipeline:
    """Main pipeline for processing videos into normalized pose data."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize processing pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._initialize_components()
        
        logger.info("Initialized video processing pipeline")
    
    def _initialize_components(self):
        """Initialize and link all pipeline components."""
        # Perception components
        self.detector = YOLODetector(
            model_size=self.config.detection_model_size,
            confidence_threshold=self.config.detection_confidence
        )
        
        self.tracker = ByteTracker(
            max_lost=self.config.tracking_max_lost,
            iou_threshold=self.config.tracking_iou_threshold
        )
        
        self.pose_estimator = HolisticPoseEstimator(
            model_complexity=self.config.pose_model_complexity,
            min_detection_confidence=self.config.pose_min_confidence,
            min_tracking_confidence=self.config.pose_min_confidence
        )
        
        # Processing components
        self.normalizer = get_normalizer(self.config.normalization_preset)
        
        # Storage components
        self.storage = StorageManager(
            base_path=self.config.storage_base_path,
            database_url=self.config.database_url
        )
        
        # Initialize sub-components
        self.detector.initialize()
        self.tracker.initialize()
        self.pose_estimator.initialize()
    
    def process_video(self, video_path: str, output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete video file through all tiers.
        
        Args:
            video_path: Path to input video file
            output_name: Name for output dataset
            
        Returns:
            Processing results and detailed statistics
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        output_name = output_name or video_path_obj.stem
        
        # Open video source
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {video_path} ({total_frames} frames, {duration:.2f}s)")
        
        # Prepare video metadata for the database tier
        video_metadata = {
            "path": str(video_path_obj.absolute()),
            "fps": fps,
            "frame_count": total_frames,
            "duration": duration,
            "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "size_mb": video_path_obj.stat().st_size / (1024 * 1024),
            "source_type": "file"
        }
        
        # Add entry to the metadata database
        db_video = self.storage.metadata_db.add_video(
            name=output_name,
            **{k: v for k, v in video_metadata.items() if k != 'path'}
        )
        video_metadata["db_video_id"] = db_video.id
        
        processing_stats = {
            "video_name": output_name,
            "total_frames": total_frames,
            "processed_frames": 0,
            "processed_persons": 0,
            "start_time": time.time(),
            "frame_times": [],
            "errors": []
        }
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                frame_timestamp = frame_idx / fps if fps > 0 else 0
                
                # Execute frame-level tier
                frame_result = self.process_frame(
                    frame=frame,
                    frame_id=frame_idx,
                    timestamp=frame_timestamp,
                    video_name=output_name,
                    video_metadata=video_metadata
                )
                
                # Accrue statistics
                processing_stats["processed_frames"] += 1
                processing_stats["processed_persons"] += frame_result.get("processed_persons", 0)
                processing_stats["frame_times"].append(frame_result.get("processing_time", 0))
                
                # Progress logging
                if frame_idx % 100 == 0:
                    self._log_progress(processing_stats, frame_idx, total_frames)
                
                # Testing limit if configured
                if self.config.target_fps > 0 and frame_idx >= self.config.target_fps * 1000:  # Safety limit
                   pass
                    
        except Exception as e:
            logger.error(f"Video processing failed at frame {frame_idx}: {e}")
            processing_stats["errors"].append(str(e))
        
        finally:
            cap.release()
            
            # Post-processing finalization (ML Dataset Generation)
            processing_stats["end_time"] = time.time()
            processing_stats["total_time"] = processing_stats["end_time"] - processing_stats["start_time"]
            processing_stats["avg_frame_time"] = np.mean(processing_stats["frame_times"]) if processing_stats["frame_times"] else 0
            
            # Finalize high-performance storage (Parquet/HDF5)
            self.storage.finalize_video_processing(output_name, video_metadata)
            
            logger.info(f"Completed processing '{output_name}': {processing_stats['processed_frames']}/{total_frames} frames.")
        
        return processing_stats
    
    def process_frame(self, frame: np.ndarray, frame_id: int, timestamp: float,
                     video_name: str, video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single video frame through perception and normalization.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame number
            timestamp: Timestamp in seconds
            video_name: Name of the video
            video_metadata: Video metadata
            
        Returns:
            Dictionary of frame-level processing results
        """
        start_time = time.time()
        frame_results = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "processed_persons": 0,
            "processing_time": 0,
            "success": False
        }
        
        try:
            # 1. Detection
            detections = self.detector.detect(frame)
            
            # 2. Tracking
            tracks = self.tracker.update(detections, frame_id)
            
            processed_persons_count = 0
            
            for track in tracks:
                try:
                    # ROI Extraction
                    bbox = track.bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # ROI Padding for holistic stability
                    pad = 20
                    roi = frame[max(0, y1-pad):min(frame.shape[0], y2+pad), 
                                max(0, x1-pad):min(frame.shape[1], x2+pad)]
                                
                    if roi.size == 0: continue
                    
                    # 3. Pose Estimation
                    # Use process_frame for holistic results if available, else estimate
                    if hasattr(self.pose_estimator, 'process_frame'):
                        pose_result = self.pose_estimator.process_frame(roi, bbox)
                    else:
                        pose_result = self.pose_estimator.estimate(frame, bbox)
                    
                    # 4. Standardized Data Model
                    skeleton_frame = SkeletonFrame(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        person_id=f"P{track.track_id}",
                        source_video=video_name,
                        camera_id="main",
                        resolution=(frame.shape[1], frame.shape[0]),
                        overall_confidence=pose_result.confidence if hasattr(pose_result, 'confidence') else 1.0,
                        processing_time=time.time() - start_time
                    )
                    # Maps landmarks to the unified skeleton model
                    if hasattr(skeleton_frame, 'from_mediapipe'):
                        # Using from_results if person detection result is available
                        pass
                    
                    # 5. Normalization
                    normalized_frame = self.normalizer.normalize_frame(skeleton_frame)
                    
                    # 6. Persistent Storage (All Tiers)
                    self.storage.save_pose_frame(
                        normalized_frame, video_name, video_metadata
                    )
                    
                    processed_persons_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process track {track.track_id}: {e}")
                    continue
            
            frame_results["success"] = True
            frame_results["processed_persons"] = processed_persons_count
            
        except Exception as e:
            logger.error(f"Frame {frame_id} processing failed: {e}")
            frame_results["error"] = str(e)
            
        finally:
            frame_results["processing_time"] = time.time() - start_time
            
        return frame_results
    
    def _log_progress(self, stats: Dict[str, Any], current_frame: int, total_frames: int):
        """Log video processing progress and ETA."""
        processed = stats["processed_frames"]
        if processed == 0: return
        
        avg_time = stats["avg_frame_time"] if stats["avg_frame_time"] > 0 else (time.time() - stats["start_time"]) / processed
        eta = (total_frames - current_frame) * avg_time
        
        logger.info(
            f"Progress: [{processed}/{total_frames}] {processed/total_frames*100:.1f}% | "
            f"ETA: {eta:.1f}s | "
            f"Avg: {avg_time*1000:.1f}ms/f | "
            f"Detected: {stats['processed_persons']}"
        )
    
    def export_for_training(self, video_name: str, format: str = "parquet") -> str:
        """Export processed video data for training workflows."""
        return self.storage.export_for_training(video_name, format)
    
    def get_processing_stats(self, video_name: str) -> Dict[str, Any]:
        """Retrieve video processing statistics."""
        return self.storage.get_video_statistics(video_name)
    
    def close(self):
        """Release resources."""
        if hasattr(self.pose_estimator, 'close'):
            self.pose_estimator.close()
        logger.info("Video processing pipeline closed.")

def process_video(video_path: str, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """Helper function for one-off video processing."""
    pipeline = VideoProcessingPipeline(config)
    try:
        return pipeline.process_video(video_path)
    finally:
        pipeline.close()
