from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid
import os
from pathlib import Path
from loguru import logger

from core.settings import settings
from ..services.dataset_builder import dataset_builder

router = APIRouter()

class DatasetBuildRequest(BaseModel):
    max_videos: Optional[int] = 20
    name: Optional[str] = None
    resume: Optional[bool] = False

# In-memory job state (in-memory for now, use Redis or a database in production)
build_jobs: Dict[str, Dict[str, Any]] = {}

@router.post("/build", status_code=status.HTTP_202_ACCEPTED)
async def start_dataset_build(
    request: DatasetBuildRequest,
    background_tasks: BackgroundTasks
):
    """
    Start an asynchronous automated dataset construction job.
    Returns a job_id for state tracking.
    """
    job_id = str(uuid.uuid4())
    
    # Register job state
    build_jobs[job_id] = {
        "job_id": job_id,
        "status": "starting",
        "progress": 0,
        "result": None,
        "error": None
    }
    
    # Dispatch to background
    background_tasks.add_task(run_build_task, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "started",
        "detail": "Dataset construction initiated."
    }

@router.get("/status/{job_id}")
async def get_build_status(job_id: str):
    """Monitor the progress of a dataset build job."""
    if job_id not in build_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return build_jobs[job_id]

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_datasets():
    """List all completed datasets with their summaries."""
    return dataset_builder.list_datasets()

@router.get("/history")
async def get_dataset_history():
    """Alias for listing dataset history."""
    datasets = dataset_builder.list_datasets()
    return {"datasets": datasets}

@router.get("/{dataset_id}")
async def get_dataset_details(dataset_id: str):
    """Retrieve detailed reports for a specific dataset."""
    dataset_path = settings.DATA_DIR / "datasets" / dataset_id
    summary_path = dataset_path / "summary.json"
    
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Dataset not found or summary missing.")
    
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        # Also return file list if available
        files = os.listdir(dataset_path)
        
        return {
            "summary": summary,
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

async def run_build_task(job_id: str, request: DatasetBuildRequest):
    """Background task to run the DatasetBuilder."""
    try:
        build_jobs[job_id]["status"] = "processing"
        
        # Trigger the builder orchestration
        summary = await dataset_builder.build_dataset(
            max_videos=request.max_videos,
            resume_from=request.name if request.resume else None
        )
        
        build_jobs[job_id]["status"] = "completed"
        build_jobs[job_id]["result"] = summary
        build_jobs[job_id]["progress"] = 100
        
    except Exception as e:
        logger.error(f"Dataset build task {job_id} failed: {e}")
        build_jobs[job_id]["status"] = "failed"
        build_jobs[job_id]["error"] = str(e)
