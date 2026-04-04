"""
Video upload endpoints.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from typing import Optional, List
from loguru import logger
import json
from datetime import datetime

from ..schemas import VideoUploadRequest, VideoUploadResponse, UploadSource, ErrorResponse
from ..services.video_service import video_service

router = APIRouter()

@router.post(
    "/upload/video",
    response_model=VideoUploadResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to upload"),
    source: UploadSource = Form(UploadSource.UPLOAD, description="Video source type"),
    metadata: Optional[str] = Form(None, description="Optional metadata as JSON string")
):
    """
    Upload a video file for processing.
    
    Supports various video formats and sources.
    """
    try:
        # Parse metadata if provided
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid metadata JSON format"
                )
        
        # Save the uploaded file asynchronously
        response = await video_service.save_uploaded_video(file, source, metadata_dict)
        
        logger.info(f"Video uploaded successfully: {response.filename}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process upload"
        )

@router.delete("/upload/video/{video_id}", response_model=dict)
async def delete_video(
    video_id: str,
    source: UploadSource = UploadSource.UPLOAD
):
    """
    Delete an uploaded video file.
    
    Useful for cleaning up processed videos.
    """
    success = video_service.delete_video(video_id, source)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found"
        )
    
    return {
        "message": "Video deleted successfully",
        "video_id": video_id,
        "deleted": True
    }

@router.get("/upload/videos", response_model=List[VideoUploadResponse])
async def list_uploaded_videos(
    source: Optional[UploadSource] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all uploaded videos with optional filtering.
    
    Returns metadata about uploaded videos.
    """
    # TODO: Implement video listing from database/persistence layer
    # This currently provides a standardized mock for UI development
    
    return [
        VideoUploadResponse(
            id="example_id",
            filename="example.mp4",
            source=UploadSource.UPLOAD,
            upload_time=datetime.now(),
            status="completed",
            message="Mock video entry for development",
            file_path="/app/data/raw/uploads/example.mp4"
        )
    ]
