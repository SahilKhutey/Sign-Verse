"""
Unified storage manager that coordinates all storage tiers in the SignVerse ecosystem.
Provides a single interface for File Storage, Metadata Database, and High-Performance Storage.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from .file_storage import FileStorage
from .metadata_db import MetadataDB
from .high_performance_storage import HighPerformanceStorage
from ..data_models.skeleton import SkeletonFrame
from ..data_models.trajectory import PersonTrajectory, TemporalTracker

class StorageManager:
    """Manages all storage tiers in a unified interface."""
    
    def __init__(self, 
                 base_path: str = "./data",
                 database_url: str = "sqlite:///./data/metadata.db"):
        """
        Initialize storage manager.
        
        Args:
            base_path: Base path for file storage
            database_url: Database connection URL
        """
        self.file_storage = FileStorage(base_path)
        self.metadata_db = MetadataDB(database_url)
        # High-performance storage is typically located within the processed directory
        self.hp_storage = HighPerformanceStorage(str(Path(base_path) / "processed"))
        self.temporal_tracker = TemporalTracker()
        
        logger.info("Initialized unified storage manager")
    
    def save_pose_frame(self, frame: SkeletonFrame, video_name: str, 
                       video_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save pose frame to all storage tiers.
        
        Args:
            frame: SkeletonFrame to save
            video_name: Source video name
            video_metadata: Additional video metadata (path, duration, fps, etc.)
            
        Returns:
            Storage results across tiers
        """
        results = {}
        
        # 1. Save to file system (Primary JSON Archival)
        json_path = self.file_storage.save_pose_frame(frame, video_name)
        results["file_system"] = json_path
        
        # 2. Update metadata database (Structured Query Layer)
        if video_metadata:
            try:
                # Ensure video exists in database
                video = self.metadata_db.add_video(
                    name=video_name,
                    path=video_metadata.get("path", ""),
                    duration=video_metadata.get("duration", 0),
                    fps=video_metadata.get("fps", 30),
                    resolution=video_metadata.get("resolution", "1920x1080"),
                    frame_count=video_metadata.get("frame_count", 0),
                    size_mb=video_metadata.get("size_mb", 0),
                    source_type=video_metadata.get("source_type", "upload")
                )
                
                # Add frame metadata
                frame_meta = self.metadata_db.add_frame(
                    video_id=video.id,
                    frame_number=frame.frame_id,
                    timestamp=frame.timestamp,
                    pose_file_path=json_path,
                    person_count=1,  # Simple assumption of 1 person; can be updated based on pipeline results
                    processing_time=frame.processing_time or 0.0
                )
                
                # Add person metadata (Tracking info)
                person_meta = self.metadata_db.add_person(
                    video_id=video.id,
                    frame_id=frame_meta.id,
                    person_id=frame.person_id,
                    track_id=int(frame.person_id[1:]) if frame.person_id.startswith('P') and frame.person_id[1:].isdigit() else 0,
                    bbox=[0, 0, 1, 1],  # Placeholder for actual detection bbox
                    confidence=frame.overall_confidence,
                    pose_data=frame.to_dict()
                )
                
                results["metadata_db"] = {
                    "video_id": video.id,
                    "frame_id": frame_meta.id,
                    "person_id": person_meta.id
                }
            except Exception as e:
                logger.error(f"Failed to update metadata database during frame save: {e}")
        
        # 3. Update temporal tracker (Real-time Longitudinal Analysis)
        self.temporal_tracker.update(frame)
        
        return results
    
    def finalize_video_processing(self, video_name: str, video_metadata: Dict[str, Any]):
        """
        Finalize video processing and create high-performance datasets (Parquet/HDF5).
        
        Args:
            video_name: Video name
            video_metadata: Video metadata including database video_id
        """
        # Get all trajectories for this video from the temporal tracker
        trajectories = list(self.temporal_tracker.get_all_trajectories().values())
        
        if not trajectories:
            logger.warning(f"No trajectories found to finalize for video: {video_name}")
            return
            
        # Create high-performance datasets for ML training
        dataset_results = self.hp_storage.create_ml_ready_dataset(
            trajectories=trajectories,
            dataset_name=video_name,
            video_name=video_name,
            formats=["parquet", "hdf5"]
        )
        
        # Add trajectory metadata to the structured database
        for trajectory in trajectories:
            try:
                self.metadata_db.add_trajectory(
                    video_id=video_metadata.get("db_video_id", 1),
                    person_id=trajectory.person_id,
                    start_frame=trajectory.start_frame,
                    end_frame=trajectory.end_frame,
                    duration=trajectory.trajectory[-1].timestamp - trajectory.trajectory[0].timestamp if trajectory.trajectory else 0,
                    frame_count=len(trajectory.trajectory),
                    avg_confidence=np.mean([p.confidence for p in trajectory.trajectory]) if trajectory.trajectory else 0.8,
                    min_confidence=np.min([p.confidence for p in trajectory.trajectory]) if trajectory.trajectory else 0.6,
                    max_confidence=np.max([p.confidence for p in trajectory.trajectory]) if trajectory.trajectory else 0.95,
                    joint_list=list(trajectory.trajectory[0].joints.keys()) if trajectory.trajectory else [],
                    trajectory_path=dataset_results.get("parquet", "")
                )
            except Exception as e:
                logger.error(f"Failed to add trajectory metadata for {trajectory.person_id}: {e}")
        
        logger.info(f"Finalized video processing for {video_name}: {len(trajectories)} trajectories stored.")
    
    def query_poses(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query pose data across storage tiers (Database -> File System).
        
        Args:
            query: Query parameters (person_id, video_name, min_confidence)
            
        Returns:
            List of hydrated pose dictionaries
        """
        results = []
        
        # Query metadata database for relevant frames
        if "person_id" in query:
            frames = self.metadata_db.query_frames_with_person(
                query["person_id"],
                query.get("video_name"),
                query.get("min_confidence", 0.5)
            )
            
            # Load actual pose data from the file system tier
            for frame in frames:
                pose_frame = self.file_storage.load_pose_frame(
                    query.get("video_name", ""), 
                    frame.frame_number
                )
                if pose_frame:
                    results.append(pose_frame.to_dict())
        
        return results
    
    def get_video_statistics(self, video_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a video across file and database tiers.
        
        Args:
            video_name: Video name
            
        Returns:
            Aggregated video statistics
        """
        # Get file system inventory
        file_stats = {
            "pose_files_count": self.file_storage.get_video_pose_count(video_name),
            "pose_files_list": self.file_storage.list_video_poses(video_name)
        }
        
        # Get structured database statistics
        db_stats = self.metadata_db.get_video_stats(video_name)
        
        return {
            "file_system": file_stats,
            "database": db_stats,
            "video_name": video_name
        }
    
    def export_for_training(self, video_name: str, format: str = "parquet") -> str:
        """
        Export longitudinal data for machine learning training.
        
        Args:
            video_name: Video name
            format: Export format (parquet, hdf5, tfrecords)
            
        Returns:
            Path to exported dataset
        """
        # Get trajectories from the current tracking session
        trajectories = list(self.temporal_tracker.get_all_trajectories().values())
        
        if not trajectories:
            raise ValueError(f"No trajectories available for export: {video_name}")
            
        if format == "parquet":
            return self.hp_storage.save_parquet_dataset(trajectories, video_name, video_name)
        elif format == "hdf5":
            return self.hp_storage.save_hdf5_dataset(trajectories, video_name, video_name)
        elif format == "tfrecords":
            return self.hp_storage.convert_to_tfrecords(video_name, video_name)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def close(self):
        """Shutdown all storage tiers and release resources."""
        if hasattr(self, 'metadata_db'):
            self.metadata_db.close()
        logger.info("Unified storage manager shut down")
