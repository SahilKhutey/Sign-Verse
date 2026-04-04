"""
Video processing service for API.
"""
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import UploadFile, HTTPException, status
from loguru import logger
import aiofiles
import uuid
from datetime import datetime

from configs import get_config
from core.data_manager import data_manager, DataSource
from ..schemas import VideoUploadResponse, UploadSource

class VideoService:
    """Handles video upload and management."""
    
    def __init__(self):
        self.config = get_config()
    
    async def save_uploaded_video(self, file: UploadFile, source: UploadSource, 
                                metadata: Optional[Dict[str, Any]] = None) -> VideoUploadResponse:
        """Save uploaded video file with validation."""
        # Validate file extension
        allowed_exts = self.config.api.upload.allowed_extensions
        if not any(file.filename.lower().endswith(ext) for ext in allowed_exts):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extension. Allowed: {allowed_exts}"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source.value}_{file_id}_{timestamp}{Path(file.filename).suffix}"
        
        # Get save path
        save_path = data_manager.get_video_path(
            DataSource(source.value), 
            filename
        )
        
        # Save file asynchronously
        try:
            async with aiofiles.open(save_path, 'wb') as buffer:
                while chunk := await file.read(self.config.api.upload.chunk_size):
                    await buffer.write(chunk)
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save video file"
            )
        
        # Verify file was saved
        if not save_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video file was not saved correctly"
            )
        
        logger.info(f"Video saved: {save_path}")
        
        return VideoUploadResponse(
            id=file_id,
            filename=filename,
            source=source,
            upload_time=datetime.now(),
            status="completed",
            message="Video uploaded successfully",
            file_path=str(save_path)
        )
    
    def validate_video_file(self, file_path: Path) -> bool:
        """Validate video file properties."""
        if not file_path.exists():
            return False
        
        # Check file size
        max_size = self.config.api.upload.max_file_size * 1024 * 1024
        if file_path.stat().st_size > max_size:
            return False
        
        return True
    
    def delete_video(self, video_id: str, source: UploadSource) -> bool:
        """Delete a video file."""
        try:
            # Find the video file
            video_dir = data_manager.get_video_path(DataSource(source.value), "")
            video_files = list(video_dir.glob(f"*{video_id}*"))
            
            if not video_files:
                return False
            
            # Delete the file
            for video_file in video_files:
                video_file.unlink()
                logger.info(f"Deleted video: {video_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video {video_id}: {e}")
            return False

# Global service instance
video_service = VideoService()
