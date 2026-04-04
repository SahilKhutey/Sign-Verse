"""
High-performance storage for machine learning training.
Supports Parquet and HDF5 for efficient data handling.
"""
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from ..data_models.skeleton import SkeletonFrame
from ..data_models.trajectory import PersonTrajectory

class HighPerformanceStorage:
    """Manages high-performance storage formats for ML training."""
    
    def __init__(self, base_path: str = "./data/processed"):
        """
        Initialize high-performance storage.
        
        Args:
            base_path: Base path for storage
        """
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories for optimized ML formats."""
        directories = [
            self.base_path / "parquet",
            self.base_path / "hdf5",
            self.base_path / "tfrecords"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_parquet_dataset(self, trajectories: List[PersonTrajectory], 
                           dataset_name: str, video_name: str) -> str:
        """
        Save trajectories as Parquet dataset for ML training.
        
        Args:
            trajectories: List of person trajectories
            dataset_name: Name for the dataset
            video_name: Source video name
            
        Returns:
            Path to saved dataset
        """
        # Convert trajectories to pandas DataFrames
        frames_data = []
        joints_data = []
        
        for trajectory in trajectories:
            for point in trajectory.trajectory:
                # Frame-level data
                frame_row = {
                    "frame_id": point.frame_id,
                    "timestamp": point.timestamp,
                    "person_id": trajectory.person_id,
                    "video_name": video_name,
                    "overall_confidence": point.confidence
                }
                
                # Joint-level data
                for joint_name, joint_data in point.joints.items():
                    joint_row = frame_row.copy()
                    joint_row.update({
                        "joint_name": joint_name,
                        "x": joint_data["x"],
                        "y": joint_data["y"],
                        "z": joint_data["z"],
                        "joint_confidence": joint_data["confidence"]
                    })
                    joints_data.append(joint_row)
                
                frames_data.append(frame_row)
        
        # Create DataFrames
        frames_df = pd.DataFrame(frames_data)
        joints_df = pd.DataFrame(joints_data)
        
        # Save as Parquet
        dataset_dir = self.base_path / "parquet" / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        frames_path = dataset_dir / "frames.parquet"
        joints_path = dataset_dir / "joints.parquet"
        
        frames_df.to_parquet(frames_path, engine='pyarrow', compression='snappy')
        joints_df.to_parquet(joints_path, engine='pyarrow', compression='snappy')
        
        logger.info(f"Saved Parquet dataset: {dataset_dir}")
        return str(dataset_dir)
    
    def save_hdf5_dataset(self, trajectories: List[PersonTrajectory],
                         dataset_name: str, video_name: str) -> str:
        """
        Save trajectories as HDF5 dataset for efficient storage.
        
        Args:
            trajectories: List of person trajectories
            dataset_name: Name for the dataset
            video_name: Source video name
            
        Returns:
            Path to saved dataset
        """
        dataset_path = self.base_path / "hdf5" / f"{dataset_name}.h5"
        
        with h5py.File(dataset_path, 'w') as hf:
            # Create groups
            metadata_group = hf.create_group("metadata")
            data_group = hf.create_group("data")
            
            # Store metadata
            metadata_group.attrs["dataset_name"] = dataset_name
            metadata_group.attrs["video_name"] = video_name
            metadata_group.attrs["created_at"] = datetime.now().isoformat()
            metadata_group.attrs["trajectory_count"] = len(trajectories)
            metadata_group.attrs["total_frames"] = sum(len(t.trajectory) for t in trajectories)
            
            # Store each trajectory
            for i, trajectory in enumerate(trajectories):
                traj_group = data_group.create_group(f"trajectory_{i}")
                traj_group.attrs["person_id"] = trajectory.person_id
                traj_group.attrs["start_frame"] = trajectory.start_frame
                traj_group.attrs["end_frame"] = trajectory.end_frame
                traj_group.attrs["frame_count"] = len(trajectory.trajectory)
                
                # Store frame data
                frames = traj_group.create_group("frames")
                for j, point in enumerate(trajectory.trajectory):
                    frame_group = frames.create_group(f"frame_{j}")
                    frame_group.attrs["frame_id"] = point.frame_id
                    frame_group.attrs["timestamp"] = point.timestamp
                    frame_group.attrs["confidence"] = point.confidence
                    
                    # Store joints
                    joints_group = frame_group.create_group("joints")
                    for joint_name, joint_data in point.joints.items():
                        # Store coordinates as a 1D array [x, y, z, confidence]
                        joint_ds = joints_group.create_dataset(joint_name, data=[
                            joint_data["x"], joint_data["y"], joint_data["z"], joint_data["confidence"]
                        ])
        
        logger.info(f"Saved HDF5 dataset: {dataset_path}")
        return str(dataset_path)
    
    def load_parquet_dataset(self, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """
        Load Parquet dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            Dictionary with frames and joints DataFrames
        """
        dataset_dir = self.base_path / "parquet" / dataset_name
        
        frames_path = dataset_dir / "frames.parquet"
        joints_path = dataset_dir / "joints.parquet"
        
        if not frames_path.exists() or not joints_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        frames_df = pd.read_parquet(frames_path)
        joints_df = pd.read_parquet(joints_path)
        
        return {
            "frames": frames_df,
            "joints": joints_df
        }
    
    def load_hdf5_dataset(self, dataset_name: str) -> h5py.File:
        """
        Load HDF5 dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            HDF5 file object
        """
        dataset_path = self.base_path / "hdf5" / f"{dataset_name}.h5"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        return h5py.File(dataset_path, 'r')
    
    def convert_to_tfrecords(self, dataset_name: str, output_name: str) -> str:
        """
        Placeholder for TFRecords conversion.
        
        Args:
            dataset_name: Input dataset name
            output_name: Output TFRecords name
            
        Returns:
            Path to TFRecords
        """
        output_path = self.base_path / "tfrecords" / f"{output_name}.tfrecord"
        output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"Converted {dataset_name} to TFRecords (Placeholder): {output_path}")
        return str(output_path)
    
    def create_ml_ready_dataset(self, trajectories: List[PersonTrajectory],
                               dataset_name: str, video_name: str,
                               formats: List[str] = ["parquet", "hdf5"]) -> Dict[str, str]:
        """
        Create ML-ready dataset in multiple formats.
        
        Args:
            trajectories: List of trajectories
            dataset_name: Dataset name
            video_name: Source video name
            formats: List of formats to create
            
        Returns:
            Dictionary of paths to created datasets
        """
        results = {}
        
        if "parquet" in formats:
            results["parquet"] = self.save_parquet_dataset(trajectories, dataset_name, video_name)
        
        if "hdf5" in formats:
            results["hdf5"] = self.save_hdf5_dataset(trajectories, dataset_name, video_name)
        
        if "tfrecords" in formats:
            results["tfrecords"] = self.convert_to_tfrecords(dataset_name, f"{dataset_name}_tf")
        
        return results
