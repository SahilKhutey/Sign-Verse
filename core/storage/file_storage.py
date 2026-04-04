"""
File system storage for raw and processed data.
Supports archival of videos, pose data (JSON), and exports.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import pandas as pd

from ..data_models.skeleton import SkeletonFrame
from ..data_models.storage_schema import PoseStorageSchema

class FileStorage:
    """Manages file system storage for pose data and videos."""
    
    def __init__(self, base_path: str = "./data"):
        """
        Initialize file storage.
        
        Args:
            base_path: Base path for data storage
        """
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories for tiered storage."""
        directories = [
            self.base_path / "raw" / "videos",
            self.base_path / "raw" / "youtube",
            self.base_path / "processed" / "poses",
            self.base_path / "processed" / "annotations",
            self.base_path / "processed" / "parquet",
            self.base_path / "processed" / "hdf5",
            self.base_path / "exports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {directory}")
    
    def save_pose_frame(self, frame: SkeletonFrame, video_name: str) -> str:
        """
        Save pose frame as JSON file using the PoseStorageSchema.
        
        Args:
            frame: SkeletonFrame to save
            video_name: Source video name
            
        Returns:
            Path to saved file
        """
        # Create video-specific directory
        video_dir = self.base_path / "processed" / "poses" / video_name
        video_dir.mkdir(exist_ok=True)
        
        # Generate filename
        filename = f"frame_{frame.frame_id:06d}.json"
        filepath = video_dir / filename
        
        # Convert to storage schema for standardized persistence
        storage_schema = PoseStorageSchema.from_skeleton_frame(frame)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            # Use Pydantic's dict() or json()
            json.dump(storage_schema.dict(), f, indent=2, default=str)
        
        logger.debug(f"Saved pose frame: {filepath}")
        return str(filepath)
    
    def load_pose_frame(self, video_name: str, frame_id: int) -> Optional[SkeletonFrame]:
        """
        Load pose frame from JSON file.
        
        Args:
            video_name: Source video name
            frame_id: Frame number
            
        Returns:
            Loaded SkeletonFrame or None
        """
        filepath = self.base_path / "processed" / "poses" / video_name / f"frame_{frame_id:06d}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert back to SkeletonFrame via Pydantic unpacking
            return SkeletonFrame(**data)
        except Exception as e:
            logger.error(f"Failed to load pose frame {filepath}: {e}")
            return None
    
    def save_video_file(self, video_file, video_name: str) -> str:
        """
        Save uploaded video file to raw storage.
        
        Args:
            video_file: Video file-like object or path
            video_name: Name for the video (without extension)
            
        Returns:
            Path to saved video
        """
        video_dir = self.base_path / "raw" / "videos"
        video_path = video_dir / f"{video_name}.mp4"
        
        # Handle both paths and file-like objects
        if isinstance(video_file, (str, Path)):
            shutil.copy2(video_file, video_path)
        else:
            with open(video_path, 'wb') as f:
                if hasattr(video_file, 'read'):
                    f.write(video_file.read())
                else:
                    shutil.copyfileobj(video_file, f)
        
        logger.info(f"Saved video: {video_path}")
        return str(video_path)
    
    def list_video_poses(self, video_name: str) -> List[str]:
        """
        List all pose files for a specific video.
        
        Args:
            video_name: Video name
            
        Returns:
            List of pose file paths
        """
        video_dir = self.base_path / "processed" / "poses" / video_name
        if not video_dir.exists():
            return []
        
        return sorted([str(f) for f in video_dir.glob("frame_*.json")])
    
    def get_video_pose_count(self, video_name: str) -> int:
        """
        Get number of processed frames for a video.
        
        Args:
            video_name: Video name
            
        Returns:
            Number of pose files
        """
        video_dir = self.base_path / "processed" / "poses" / video_name
        if not video_dir.exists():
            return 0
        
        return len(list(video_dir.glob("frame_*.json")))
