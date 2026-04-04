"""
Main pipeline orchestration and execution.
"""
from typing import Dict, Any, List, Optional
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from datetime import datetime
import importlib

from configs import get_config
from core.data_manager import data_manager

@flow(
    name="full_processing_pipeline",
    description="Complete pipeline from raw video to labeled poses",
    task_runner=ConcurrentTaskRunner(),
    retries=2,
    retry_delay_seconds=30,
    timeout_seconds=3600
)
def run_full_pipeline(video_path: str, source: str = "upload") -> Dict[str, Any]:
    """
    Execute the complete processing pipeline.
    
    Args:
        video_path: Path to the video file
        source: Data source type ('upload', 'youtube')
        
    Returns:
        Pipeline execution results
    """
    logger = get_run_logger()
    logger.info(f"Starting full pipeline for: {video_path}")
    
    # Import pipeline components dynamically
    try:
        from pipelines.preprocessing.frame_extractor import extract_frames
        from pipelines.pose_estimation.mediapipe_pipeline import process_video
        # The labeling and transformation modules are to be implemented in future steps
        # from pipelines.labeling.auto_labeler import auto_label_poses
        # from pipelines.transformation.normalize_joints import normalize_pose_sequence
    except ImportError as e:
        logger.warning(f"Some pipeline components are not yet implemented: {e}")
    
    try:
        # Step 1: Extract frames
        frames = extract_frames(video_path)
        
        # Step 2: Pose estimation
        pose_data = process_video(video_path)
        
        # Step 3: Auto-labeling (Placeholder for future implementation)
        # labeled_data = auto_label_poses(pose_data)
        
        # Step 4: Normalization (Placeholder for future implementation)
        # normalized_data = normalize_pose_sequence(pose_data)
        
        logger.info(f"Pipeline completed successfully. Processed {len(frames)} frames.")
        return {
            "success": True,
            "frames_processed": len(frames),
            "poses_extracted": len(pose_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

@flow(name="quick_pose_pipeline")
def run_quick_pose_pipeline(video_path: str) -> List[Any]:
    """Quick pipeline for pose estimation only."""
    from pipelines.pose_estimation.mediapipe_pipeline import process_video
    return process_video(video_path)

def get_pipeline(pipeline_name: str):
    """Get a pipeline function by name."""
    pipelines = {
        "full_processing": run_full_pipeline,
        "quick_pose": run_quick_pose_pipeline,
    }
    return pipelines.get(pipeline_name)

def run_pipeline_from_config(pipeline_config: Dict[str, Any]):
    """Run pipeline based on configuration."""
    pass  # Implementation for config-driven pipeline execution
