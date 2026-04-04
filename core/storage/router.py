"""
Storage Tier Router for the SignVerse Storage System.
Orchestrates data flow across File System, Database, and High-Performance tiers.
"""
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from .base import BaseStorageBackend, InitializationError
from .file_system import FileSystemBackend
from .database import DatabaseBackend
from .high_performance import HighPerformanceBackend
from ..data_models.skeleton import SkeletonFrame
from ..data_models.trajectory import PersonTrajectory

class StorageTierRouter:
    """Orchestrates data storage and retrieval across multiple storage tiers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backends: Dict[str, BaseStorageBackend] = {}
        self._setup_backends()
        
    def _setup_backends(self):
        """Initialize backends based on configuration."""
        # File System Tier (Always active by default)
        fs_config = self.config.get("file_system", {"base_path": "./data"})
        self.backends["file_system"] = FileSystemBackend(fs_config)
        
        # Database Tier (Optional)
        if "database" in self.config:
            self.backends["database"] = DatabaseBackend(self.config["database"])
            
        # High-Performance Tier (Optional)
        if "hpc" in self.config:
            self.backends["hpc"] = HighPerformanceBackend(self.config["hpc"])
            
    def initialize(self):
        """Initialize all activated backends."""
        for name, backend in self.backends.items():
            try:
                backend.initialize()
            except InitializationError as e:
                logger.error(f"Failed to initialize storage backend '{name}': {e}")
                
    def store_frame(self, frame: SkeletonFrame) -> bool:
        """Store a single skeleton frame across all active tiers."""
        success = True
        
        # 1. Store as JSON (Processed Archive)
        if "file_system" in self.backends:
            success &= self.backends["file_system"].store(frame, sub_dir="poses")
            
        # 2. Index in Database (Metadata)
        if "database" in self.backends:
            success &= self.backends["database"].store(frame)
            
        return success
        
    def store_trajectory(self, trajectory: PersonTrajectory) -> bool:
        """Store a complete person trajectory across all active tiers."""
        success = True
        
        # 1. Store as JSON (Processed Archive)
        if "file_system" in self.backends:
            success &= self.backends["file_system"].store(trajectory, sub_dir="trajectories")
            
        # 2. Index in Database (Metadata)
        if "database" in self.backends:
            success &= self.backends["database"].store(trajectory)
            
        # 3. Store high-performance format (Training Data)
        if "hpc" in self.backends:
            success &= self.backends["hpc"].store(trajectory)
            
        return success

    def store_batch(self, frames: List[SkeletonFrame], filename: str) -> bool:
        """Store a batch of frames, typically in HPC format."""
        if "hpc" in self.backends:
            return self.backends["hpc"].store(frames, filename=filename)
        return False

    def query_frames(self, person_id: str = None, video_id: str = None, limit: int = 100) -> List[Any]:
        """Query metadata database for frames."""
        if "database" in self.backends:
            from .database import FrameRecord
            filters = {}
            if person_id: filters["person_id"] = person_id
            if video_id: filters["video_id"] = video_id
            return self.backends["database"].retrieve(FrameRecord, filters=filters, limit=limit)
        return []

    def get_backend(self, name: str) -> Optional[BaseStorageBackend]:
        """Get a specific storage backend by name."""
        return self.backends.get(name)
