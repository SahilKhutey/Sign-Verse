"""
Centralized data management utilities.
Handles file operations, validation, and dataset creation.
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from .data_models import PoseData, DatasetManifest, DataSource
from configs import get_config

class DataManager:
    """Manages data operations across the entire system."""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.config = get_config()
        self.base_path = base_path or Path(self.config.data.base_path)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all necessary data directories if they don't exist."""
        directories = [
            self.base_path / 'raw' / 'uploads',
            self.base_path / 'raw' / 'youtube_videos',
            self.base_path / 'processed' / 'frames',
            self.base_path / 'processed' / 'poses',
            self.base_path / 'processed' / 'annotations',
            self.base_path / 'labeled' / 'actions',
            self.base_path / 'labeled' / 'joints',
            self.base_path / 'datasets',
            self.base_path / 'metadata'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def get_video_path(self, source: DataSource, filename: str) -> Path:
        """Get the full path for a video file based on its source."""
        if source == DataSource.UPLOAD:
            return self.base_path / 'raw' / 'uploads' / filename
        elif source == DataSource.YOUTUBE:
            return self.base_path / 'raw' / 'youtube_videos' / filename
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def save_pose_data(self, pose_data: PoseData, output_filename: str) -> Path:
        """
        Save pose data to processed directory with validation.
        
        Args:
            pose_data: Validated PoseData object
            output_filename: Output filename (without path)
            
        Returns:
            Path to the saved file
        """
        output_path = self.base_path / 'processed' / 'poses' / output_filename
        
        # Validate against schema
        try:
            pose_data_dict = pose_data.dict()
        except Exception as e:
            logger.error(f"Pose data validation failed: {e}")
            raise
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(pose_data_dict, f, indent=2)
        
        logger.info(f"Saved pose data to {output_path}")
        return output_path
    
    def load_pose_data(self, filename: str) -> PoseData:
        """Load and validate pose data from file."""
        file_path = self.base_path / 'processed' / 'poses' / filename
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return PoseData(**data)
    
    def create_dataset_manifest(self, manifest: DatasetManifest) -> Path:
        """Create a dataset manifest file."""
        dataset_dir = self.base_path / 'datasets' / f"{manifest.name}_v{manifest.version}"
        dataset_dir.mkdir(exist_ok=True)
        
        manifest_path = dataset_dir / 'dataset.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest.dict(), f, indent=2)
        
        logger.info(f"Created dataset manifest: {manifest_path}")
        return manifest_path

# Global data manager instance
data_manager = DataManager()
