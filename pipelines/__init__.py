"""
Pipeline core module for SignVerse System.
Central orchestration and workflow management.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

from configs import get_config
from core.data_manager import data_manager
from core.data_models import DataSource

class PipelineManager:
    """Manages pipeline execution and configuration."""
    
    def __init__(self):
        self.config = get_config()
        self.initialized = False
    
    def initialize(self):
        """Initialize pipeline components."""
        if self.initialized:
            return
            
        # Create pipeline output directories
        output_dirs = [
            self.config.pipelines.ingestion.youtube.output_dir,
            self.config.pipelines.preprocessing.frame_extraction.output_dir,
            self.config.pipelines.pose_estimation.mediapipe.output_dir,
            self.config.pipelines.labeling.auto_labeler.output_dir,
            self.config.pipelines.transformation.normalization.output_dir
        ]
        
        for directory in output_dirs:
            # We use string interpolation from Dynaconf if needed, but here we expect absolute paths or relative to root
            # However, Dynaconf should have already resolved ${DATA_BASE_PATH} if it's set in env.
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.initialized = True
        logger.info("Pipeline manager initialized")

# Global pipeline manager
pipeline_manager = PipelineManager()
