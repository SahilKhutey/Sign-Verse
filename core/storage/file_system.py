"""
File system storage backend for the SignVerse Storage System.
Supports local archival of raw videos, JSON pose data, and annotations.
"""
import os
import json
import shutil
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from loguru import logger

from .base import BaseStorageBackend, InitializationError
from ..data_models.skeleton import SkeletonFrame
from ..data_models.trajectory import PersonTrajectory

class FileSystemBackend(BaseStorageBackend):
    """Handles storage of raw and processed files on the local filesystem."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_path = Path(config.get("base_path", "./data"))
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.annotation_path = self.base_path / "annotations"
        
    def initialize(self):
        """Create necessary directory structure."""
        try:
            for path in [self.raw_path, self.processed_path, self.annotation_path]:
                path.mkdir(parents=True, exist_ok=True)
            self.initialized = True
            logger.info(f"FileSystemBackend initialized at {self.base_path}")
        except Exception as e:
            raise InitializationError(f"Failed to create directories: {e}")
            
    def store(self, data: Union[SkeletonFrame, PersonTrajectory, Dict[str, Any]], 
              sub_dir: str = "poses", 
              filename: Optional[str] = None) -> bool:
        """
        Store data as JSON in the processed directory.
        
        Args:
            data: Data object to store (SkeletonFrame, PersonTrajectory, or Dict)
            sub_dir: Subdirectory within 'processed' (e.g., 'poses', 'trajectories')
            filename: Optional filename. If None, generated from data attributes.
        """
        if not self.initialized:
            self.initialize()
            
        try:
            save_dir = self.processed_path / sub_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine filename and serialize data
            if isinstance(data, SkeletonFrame):
                output_name = filename or f"pose_{data.person_id}_{data.frame_id:06d}.json"
                content = data.dict()
            elif isinstance(data, PersonTrajectory):
                output_name = filename or f"trajectory_{data.person_id}_{data.start_frame}_{data.end_frame}.json"
                # Custom serialization for trajectory points
                content = {
                    "person_id": data.person_id,
                    "start_frame": data.start_frame,
                    "end_frame": data.end_frame,
                    "points": [p.__dict__ for p in data.trajectory],
                    "metadata": data.metadata
                }
            else:
                if not filename:
                    raise ValueError("Filename must be provided for generic dictionary data")
                output_name = filename
                content = data
                
            file_path = save_dir / output_name
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2, default=str)
                
            return True
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            return False
            
    def retrieve(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Retrieve and parse a JSON file."""
        path = Path(file_path)
        if not path.exists():
            return None
            
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_path}: {e}")
            return None
            
    def delete(self, file_path: Union[str, Path]) -> bool:
        """Delete a file from the filesystem."""
        path = Path(file_path)
        try:
            if path.is_file():
                os.remove(path)
            elif path.is_dir():
                shutil.rmtree(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False

    def store_raw(self, source_path: Union[str, Path], destination_name: str) -> bool:
        """Copy a raw video file to the raw storage directory."""
        if not self.initialized:
            self.initialize()
        
        try:
            dest_path = self.raw_path / destination_name
            shutil.copy2(source_path, dest_path)
            return True
        except Exception as e:
            logger.error(f"Failed to store raw file: {e}")
            return False
