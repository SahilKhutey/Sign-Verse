from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import redis
import json
import uuid
import os
from loguru import logger

# Import components
from .pose_extractor import MediaPipePoseExtractor
from core.settings import settings
from configs import get_config
from core.data_manager import data_manager
from core.data_models import PoseData, Keypoint, DataSource

app = FastAPI(
    title="Pose Extraction Service", 
    version=settings.app.version,
    debug=settings.app.debug
)

# Use configured Redis settings
redis_client = redis.Redis(
    host=settings.redis.host,
    port=settings.redis.port,
    db=settings.redis.db,
    decode_responses=settings.redis.decode_responses
)

# Use Dynaconf config for model parameters
config = get_config()
pose_extractor = MediaPipePoseExtractor(
    static_image_mode=config.pose_estimation.static_image_mode,
    model_complexity=config.pose_estimation.model_complexity,
    min_detection_confidence=config.pose_estimation.min_detection_confidence
)

# Request/Response Models
class PoseExtractionRequest(BaseModel):
    video_path: str  # Filename in the uploads directory
    callback_url: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

@app.post("/jobs", response_model=dict)
async def create_pose_job(request: PoseExtractionRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_key = f"pose:job:{job_id}"

    # Validate the source video exists using DataManager
    video_filename = request.video_path
    video_path = data_manager.get_video_path(DataSource.UPLOAD, video_filename)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")

    initial_data = {
        "job_id": job_id,
        "status": "pending",
        "video_path": video_filename,
        "callback_url": request.callback_url
    }
    redis_client.set(job_key, json.dumps(initial_data))
    redis_client.lpush("pose:job:queue", job_id)

    logger.info(f"Pose job created", job_id=job_id, video_path=video_filename)
    background_tasks.add_task(process_job_from_queue)
    
    return {"job_id": job_id, "status": "submitted", "message": "Job queued for processing."}

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job_key = f"pose:job:{job_id}"
    job_data = redis_client.get(job_key)
    
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    
    job_info = json.loads(job_data)
    return JobStatusResponse(**job_info)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "pose-service"}

@app.get("/config")
async def show_config():
    """Endpoint to show current configuration (for debugging)."""
    return {
        "redis": {
            "host": settings.redis.host,
            "port": settings.redis.port
        },
        "pose_model": {
            "model_complexity": settings.pose_model.model_complexity,
            "min_detection_confidence": settings.pose_model.min_detection_confidence
        }
    }

# --- Background Task Logic ---
async def process_job_from_queue():
    """Pops a job ID from the queue and processes it."""
    try:
        job_result = redis_client.brpop("pose:job:queue", timeout=1)
        if job_result:
            _, job_id = job_result
            await process_single_job(job_id)
    except Exception as e:
        logger.error(f"Error in queue processing: {e}")

async def process_single_job(job_id: str):
    job_key = f"pose:job:{job_id}"
    job_data = redis_client.get(job_key)
    if not job_data:
        logger.warning(f"Job data missing for {job_id}. Skipping.")
        return

    job_info = json.loads(job_data)
    job_info["status"] = "processing"
    redis_client.set(job_key, json.dumps(job_info))

    video_filename = job_info["video_path"]
    
    try:
        # Use data manager to get the full path
        video_path = data_manager.get_video_path(DataSource.UPLOAD, video_filename)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Starting processing for job {job_id}", path=video_path)
        results = pose_extractor.extract_from_video(str(video_path))
        
        # Convert to standardized format and save
        pose_data_list = []
        for frame_data in results:
            if not frame_data["landmarks"]:
                continue
                
            keypoints = [
                Keypoint(
                    id=idx,
                    name=config.keypoint_labels.labels[idx],
                    x=landmark["x"],
                    y=landmark["y"],
                    z=landmark["z"],
                    confidence=landmark["visibility"],
                    visible=landmark["visibility"] > config.pose_estimation.thresholds.get(config.keypoint_labels.labels[idx], 0.3)
                ) for idx, landmark in enumerate(frame_data["landmarks"])
            ]
            
            pose_data = PoseData(
                source_video=video_filename,
                frame_number=frame_data["frame_number"],
                timestamp=frame_data["frame_number"] / 30.0,
                keypoints=keypoints
            )
            
            # Save each frame's pose data as a standardized JSON file
            output_filename = f"{video_path.stem}_frame_{frame_data['frame_number']:06d}.json"
            data_manager.save_pose_data(pose_data, output_filename)
            pose_data_list.append(pose_data.dict())
        
        job_info["status"] = "completed"
        job_info["result"] = pose_data_list
        job_info["message"] = f"Successfully processed {len(pose_data_list)} frames."
        logger.success(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job_info["status"] = "failed"
        job_info["message"] = str(e)
        job_info["result"] = None

    finally:
        redis_client.set(job_key, json.dumps(job_info))
