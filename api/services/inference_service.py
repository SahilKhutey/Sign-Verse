"""
Inference service for pose estimation.
"""
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import HTTPException, status
from loguru import logger
import uuid
from datetime import datetime
import json

from configs import get_config
from core.data_manager import data_manager, DataSource
from core.data_models import PoseData
from models.inference.predict_pose import get_pose_predictor
from ..schemas import PoseEstimationRequest, PoseEstimationResponse, PoseResult

class InferenceService:
    """Handles pose estimation inference."""
    
    def __init__(self):
        self.config = get_config()
        self.pose_predictor = get_pose_predictor()
        # In-memory job tracker for simple local async status retrieval
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
    
    async def process_pose_estimation(self, request: PoseEstimationRequest) -> PoseEstimationResponse:
        """Register and start a pose estimation job asynchronously."""
        job_id = str(uuid.uuid4())
        
        # Register job information
        self.active_jobs[job_id] = {
            "request": request,
            "status": "processing",
            "start_time": datetime.now(),
            "progress": 0
        }
        
        # Fire-and-forget background task
        asyncio.create_task(self._process_job(job_id, request))
        
        return PoseEstimationResponse(
            job_id=job_id,
            video_id=request.video_id,
            status="queued",
            total_frames=0,
            processed_frames=0,
            estimated_time=None
        )
    
    async def _process_job(self, job_id: str, request: PoseEstimationRequest):
        """Internal background job handler."""
        try:
            # Locate the video file within the system's ingestion path
            video_path = self._find_video_file(request.video_id)
            if not video_path:
                raise ValueError(f"Video {request.video_id} not found across all ingestion sources.")
            
            # Explicitly update job status to processing
            self.active_jobs[job_id]["status"] = "processing"
            
            # Process video through the Pose Prediction layer
            result = await self._process_video_logic(video_path, request)
            
            # Persistent storage of results
            result_path = self._save_results(result, job_id)
            
            # Update job state to completed
            self.active_jobs[job_id].update({
                "status": "completed",
                "result_path": result_path,
                "end_time": datetime.now(),
                "progress": 100
            })
            
            logger.info(f"Inference job {job_id} successfully finalized.")
            
        except Exception as e:
            logger.error(f"Inference job {job_id} encountered a terminal error: {e}")
            if job_id in self.active_jobs:
                self.active_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": datetime.now()
                })
    
    async def _process_video_logic(self, video_path: Path, request: PoseEstimationRequest) -> PoseResult:
        """Execute extraction and refinement on the target video."""
        # TODO: Integration with MediaPipe or PoseDetector
        # This currently simulates the temporal processing of frames
        
        simulation_time = 2.0
        await asyncio.sleep(simulation_time)
        
        return PoseResult(
            video_id=request.video_id,
            frames=[],
            model_name=request.model_name,
            processing_time=simulation_time,
            resolution=(1920, 1080)
        )
    
    def _find_video_file(self, video_id: str) -> Optional[Path]:
        """Exhaustively search for video files including uploads and scrapes."""
        for source in ["upload", "youtube"]:
            video_dir = data_manager.get_video_path(DataSource(source), "")
            video_files = list(video_dir.glob(f"*{video_id}*"))
            if video_files:
                return video_files[0]
        return None
    
    def _save_results(self, result: PoseResult, job_id: str) -> str:
        """Persist pose estimation results into the processed data layer."""
        output_dir = Path(self.config.data.base_path) / "processed" / "poses"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"pose_result_{job_id}.json"
        
        with open(output_path, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        
        return str(output_path)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the current state of a registered inference job."""
        return self.active_jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Terminate or mark a running job as cancelled."""
        if job_id in self.active_jobs and self.active_jobs[job_id]["status"] == "processing":
            self.active_jobs[job_id]["status"] = "cancelled"
            return True
        return False

# Global inference service instance
inference_service = InferenceService()
