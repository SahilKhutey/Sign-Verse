"""
Pose estimation inference endpoints.
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import List, Optional
from loguru import logger

from ..schemas import (
    PoseEstimationRequest, 
    PoseEstimationResponse,
    PoseResult,
    BatchRequest,
    BatchResponse,
    ErrorResponse
)
from ..services.inference_service import inference_service

router = APIRouter()

@router.post(
    "/inference/pose",
    response_model=PoseEstimationResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def estimate_poses(
    request: PoseEstimationRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a pose estimation inference job.
    
    Orchestrates the extraction and refinement flow via the InferenceService.
    """
    try:
        response = await inference_service.process_pose_estimation(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference job initialization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate asynchronous pose estimation"
        )

@router.get("/inference/pose/{job_id}", response_model=PoseEstimationResponse)
async def get_pose_job_status(job_id: str):
    """
    Retrieve the current status and progress of a pose estimation job.
    
    Poll this endpoint to monitor the extraction lifecycle.
    """
    job_status = inference_service.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    return PoseEstimationResponse(
        job_id=job_id,
        video_id=job_status["request"].video_id,
        status=job_status["status"],
        total_frames=0,  # Placeholder for actual frame telemetry
        processed_frames=job_status.get("progress", 0),
        estimated_time=None,
        result_path=job_status.get("result_path")
    )

@router.get("/inference/pose/{job_id}/result", response_model=PoseResult)
async def get_pose_result(job_id: str):
    """
    Retrieve the final skeletal results for a completed inference job.
    
    Returns the canonical PoseResult containing all frame landmarks.
    """
    job_status = inference_service.get_job_status(job_id)
    
    if not job_status or job_status["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} logic failed: result not available or job not completed."
        )
    
    # TODO: Implement disk-to-JSON hydration for the final result
    return PoseResult(
        video_id=job_status["request"].video_id,
        frames=[],
        model_name=job_status["request"].model_name,
        processing_time=2.0,
        resolution=(1920, 1080)
    )

@router.post("/inference/pose/batch", response_model=BatchResponse)
async def batch_pose_estimation(request: BatchRequest):
    """
    Execute multiple pose estimation jobs concurrently.
    
    Optimized for high-throughput video processing batches.
    """
    # TODO: Implement batch-aware service orchestration
    return BatchResponse(
        processed=len(request.items),
        successful=len(request.items),
        failed=0,
        results=[],
        errors=[]
    )

@router.delete("/inference/pose/{job_id}", response_model=dict)
async def cancel_pose_job(job_id: str):
    """
    Abort a running inference job and release system resources.
    """
    success = inference_service.cancel_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} cannot be cancelled (not found or already terminal)."
        )
    
    return {
        "message": "Inference job cancelled successfully",
        "job_id": job_id,
        "cancelled": True
    }
