"""
High-performance storage backend for the SignVerse Storage System.
Handles optimized ML formats like Parquet, HDF5, and TFRecords for large-scale training.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from loguru import logger

from .base import BaseStorageBackend, InitializationError
from ..data_models.skeleton import SkeletonFrame
from ..data_models.trajectory import PersonTrajectory

class HighPerformanceBackend(BaseStorageBackend):
    """Handles storage of vectorized data in optimized formats (Parquet/HDF5)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(config.get("base_path", "./data/hpc"))
        self.format = config.get("format", "parquet")
        
    def initialize(self):
        """Create the base directory."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.initialized = True
            logger.info(f"HighPerformanceBackend initialized at {self.base_path} (Format: {self.format})")
        except Exception as e:
            raise InitializationError(f"Failed to create HPC directories: {e}")
            
    def store(self, data: Union[SkeletonFrame, PersonTrajectory, List[SkeletonFrame]], 
              filename: Optional[str] = None) -> bool:
        """
        Store data in a high-performance vectorized format.
        
        Args:
            data: Data object or list of objects to store.
            filename: Target filename.
        """
        if not self.initialized:
            self.initialize()
            
        try:
            if isinstance(data, (SkeletonFrame, list)):
                # Convert frame(s) to flattened DataFrame
                frames = [data] if isinstance(data, SkeletonFrame) else data
                df = self._frames_to_dataframe(frames)
                output_name = filename or f"batch_{frames[0].person_id}_{frames[0].frame_id}.parquet"
            elif isinstance(data, PersonTrajectory):
                # Convert trajectory to DataFrame
                df = self._trajectory_to_dataframe(data)
                output_name = filename or f"traj_{data.person_id}_{data.start_frame}.parquet"
            else:
                logger.warning(f"Unsupported data type for HPC storage: {type(data)}")
                return False
                
            file_path = self.base_path / output_name
            if self.format == "parquet":
                df.to_parquet(file_path, index=False, compression='snappy')
            elif self.format == "hdf5":
                df.to_hdf(file_path, key='data', mode='w')
            else:
                logger.error(f"Unsupported HPC format: {self.format}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Failed to store HPC data: {e}")
            return False
            
    def retrieve(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Retrieve data as a pandas DataFrame."""
        path = Path(file_path)
        if not path.exists():
            return None
            
        try:
            if self.format == "parquet":
                return pd.read_parquet(path)
            elif self.format == "hdf5":
                return pd.read_hdf(path, key='data')
            else:
                logger.error(f"Unsupported retrieval format: {self.format}")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve HPC data from {file_path}: {e}")
            return None
            
    def delete(self, file_path: Union[str, Path]) -> bool:
        """Delete a data file."""
        path = Path(file_path)
        try:
            if path.exists():
                os.remove(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False

    def _frames_to_dataframe(self, frames: List[SkeletonFrame]) -> pd.DataFrame:
        """Flatten multiple SkeletonFrames into a single tabular DataFrame."""
        flat_data = []
        for f in frames:
            row = {
                "frame_id": f.frame_id,
                "timestamp": f.timestamp,
                "person_id": f.person_id,
                "overall_confidence": f.overall_confidence
            }
            
            # Map body joints
            body_dict = f.body.get_joints_dict()
            for name, vec in body_dict.items():
                row[f"body_{name}_x"] = vec.x
                row[f"body_{name}_y"] = vec.y
                row[f"body_{name}_z"] = vec.z
                row[f"body_{name}_c"] = vec.confidence
            
            # Map head pose
            if f.head_pose:
                row["head_yaw"] = f.head_pose.yaw
                row["head_pitch"] = f.head_pose.pitch
                row["head_roll"] = f.head_pose.roll
                
            flat_data.append(row)
            
        return pd.DataFrame(flat_data)

    def _trajectory_to_dataframe(self, trajectory: PersonTrajectory) -> pd.DataFrame:
        """Flatten a PersonTrajectory into a tabular DataFrame."""
        flat_data = []
        for p in trajectory.trajectory:
            row = {
                "frame_id": p.frame_id,
                "timestamp": p.timestamp,
                "person_id": trajectory.person_id,
                "confidence": p.confidence
            }
            # Flatten joint dictionary
            for joint_name, pos in p.joints.items():
                row[f"{joint_name}_x"] = pos["x"]
                row[f"{joint_name}_y"] = pos["y"]
                row[f"{joint_name}_z"] = pos["z"]
                row[f"{joint_name}_c"] = pos["confidence"]
            flat_data.append(row)
            
        return pd.DataFrame(flat_data)
