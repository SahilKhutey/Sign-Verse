from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from ..services.dataset_builder import dataset_builder

router = APIRouter()

class DatasetBuildRequest(BaseModel):
    max_videos: int = 20

@router.post("/build")
async def build_dataset(
    request: DatasetBuildRequest, 
    background_tasks: BackgroundTasks
):
    """Start automated dataset building"""
    # Run in background to avoid timeout
    background_tasks.add_task(dataset_builder.build_dataset, request.max_videos)
    
    return {"status": "started", "message": "Dataset build started in background"}

@router.get("/history")
async def get_dataset_history():
    """Get dataset build history"""
    # This uses the underlying service to list datasets
    datasets = dataset_builder.list_datasets()
    return {"datasets": datasets}
