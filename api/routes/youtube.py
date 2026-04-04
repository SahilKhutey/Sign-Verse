"""
FastAPI routes for YouTube video processing pipeline.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import json
import os
import asyncio
from pathlib import Path
from loguru import logger

from ..services.youtube_pipeline import youtube_pipeline
from ..services.youtube_discovery import youtube_discovery
from ..services.video_scorer import video_scorer
from core.settings import settings

router = APIRouter()

class YouTubeProcessRequest(BaseModel):
    url: str

# In-memory job tracking (in production, use Redis or a database)
processing_jobs: Dict[str, Dict[str, Any]] = {}
discovery_jobs: Dict[str, Dict[str, Any]] = {}

class YouTubeDiscoveryRequest(BaseModel):
    max_per_category: Optional[int] = 10
    min_duration: Optional[int] = 30
    max_duration: Optional[int] = 600
    min_views: Optional[int] = 1000

class YouTubeDiscoveryProcessRequest(BaseModel):
    discovery_job_id: Optional[str] = None # If None, use latest results
    limit: Optional[int] = 5

@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def process_youtube_video(
    request: YouTubeProcessRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start an asynchronous YouTube video processing job.
    Returns a job_id for status tracking.
    """
    job_id = str(uuid.uuid4())
    
    # Register job state
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "url": request.url,
        "progress": 0,
        "result": None,
        "error": None
    }
    
    # Dispatch processing to the background
    background_tasks.add_task(run_pipeline_task, job_id, request.url)
    
    return {
        "job_id": job_id, 
        "status": "started",
        "detail": f"Processing initialized for {request.url}"
    }

@router.get("/status/{job_id}")
async def get_processing_status(job_id: str):
    """Get the current status of a processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job {job_id} not found in current session."
        )
    
    return processing_jobs[job_id]

@router.get("/results/{job_id}")
async def get_processing_results(job_id: str):
    """Retrieve full results for a completed processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Job {job_id} not found."
        )
    
    job = processing_jobs[job_id]
    if job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Job {job_id} is still {job['status']}. Please wait for completion."
        )
    
    # Load detailed results from the filesystem
    job_dir = settings.youtube.base_path / "processed" / job_id
    summary_path = job_dir / "summary.json"
    events_path = job_dir / "events.json"
    intentions_path = job_dir / "intentions.json"
    
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
        with open(events_path, "r") as f:
            events = json.load(f)
        with open(intentions_path, "r") as f:
            intentions = json.load(f)
            
        return {
            "summary": summary,
            "events": events,
            "intentions": intentions
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Processing data missing from filesystem."
        )

@router.post("/discover", status_code=status.HTTP_202_ACCEPTED)
async def start_discovery(
    request: YouTubeDiscoveryRequest,
    background_tasks: BackgroundTasks
):
    """Start a background discovery process."""
    job_id = str(uuid.uuid4())
    discovery_jobs[job_id] = {
        "job_id": job_id,
        "status": "searching",
        "timestamp": str(Path(os.getcwd()).stat().st_ctime), # Placeholder for real time
        "progress": 0,
        "results_count": 0
    }
    
    filters = {
        "min_duration": request.min_duration,
        "max_duration": request.max_duration,
        "min_views": request.min_views
    }
    
    background_tasks.add_task(run_discovery_task, job_id, request.max_per_category, filters)
    
    return {"job_id": job_id, "status": "started"}

@router.get("/discovery/status/{job_id}")
async def get_discovery_status(job_id: str):
    """Get status of a discovery job."""
    if job_id not in discovery_jobs:
        raise HTTPException(status_code=404, detail="Discovery job not found")
    return discovery_jobs[job_id]

@router.get("/discovery/results")
async def get_discovery_results():
    """Retrieve all discovered videos from persistent storage."""
    results = youtube_discovery.load_cached_results()
    return {"count": len(results), "videos": results}

@router.post("/discover/process", status_code=status.HTTP_202_ACCEPTED)
async def process_discovered_videos(
    request: YouTubeDiscoveryProcessRequest,
    background_tasks: BackgroundTasks
):
    """Queue discovered videos for processing."""
    videos = youtube_discovery.load_cached_results()
    
    if not videos:
        raise HTTPException(status_code=404, detail="No discovered videos found. Run discovery first.")
    
    # Simple selection: take the most recent ones up to the limit
    # In a real system, we might sort by view count or relevance
    to_process = videos[:request.limit]
    
    job_ids = []
    for video in to_process:
        job_id = str(uuid.uuid4())
        processing_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "url": video["url"],
            "title": video["title"],
            "progress": 0,
            "result": None,
            "error": None
        }
        background_tasks.add_task(run_pipeline_task, job_id, video["url"])
        job_ids.append(job_id)
        
    return {
        "status": "batch_started",
        "count": len(job_ids),
        "job_ids": job_ids
    }

@router.post("/discover/audit", status_code=status.HTTP_202_ACCEPTED)
async def audit_discovery_results(
    background_tasks: BackgroundTasks,
    limit: Optional[int] = 5
):
    """
    Audit discovered results by performing a visual verification and scoring.
    Updated metadata file with quality scores.
    """
    job_id = str(uuid.uuid4())
    discovery_jobs[job_id] = {
        "job_id": job_id,
        "status": "auditing",
        "progress": 0,
        "count": limit
    }
    
    background_tasks.add_task(run_audit_task, job_id, limit)
    
    return {"job_id": job_id, "status": "started", "detail": "Visual audit initiated."}

async def run_discovery_task(job_id: str, max_per_category: int, filters: Dict[str, Any]):
    """Background discovery execution."""
    try:
        results = await asyncio.to_thread(
            youtube_discovery.discover_videos, 
            max_per_category=max_per_category,
            filters=filters
        )
        discovery_jobs[job_id]["status"] = "completed"
        discovery_jobs[job_id]["results_count"] = len(results)
        discovery_jobs[job_id]["progress"] = 100
    except Exception as e:
        logger.error(f"Discovery task {job_id} failed: {e}")
        discovery_jobs[job_id]["status"] = "failed"
        discovery_jobs[job_id]["error"] = str(e)

async def run_pipeline_task(job_id: str, url: str):
    """Background task to push the job through the pipeline."""
    try:
        # Update job status to active
        processing_jobs[job_id]["status"] = "processing"
        
        # Execute the pipeline (includes concurrency limiting)
        result = await youtube_pipeline.run_pipeline(url, job_id)
        
        # Finalize job state
        if result.get("success"):
            processing_jobs[job_id]["status"] = "completed"
            processing_jobs[job_id]["result"] = result
            processing_jobs[job_id]["progress"] = 100
        else:
            processing_jobs[job_id]["status"] = "failed"
            processing_jobs[job_id]["error"] = result.get("error", "Unknown pipeline error")
            if "verification_details" in result:
                processing_jobs[job_id]["verification_details"] = result["verification_details"]
            
    except Exception as e:
        logger.error(f"Background task failed for job {job_id}: {e}")
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)

async def run_audit_task(job_id: str, limit: int):
    """Perform visual audit and scoring of discovered videos."""
    try:
        videos = youtube_discovery.load_cached_results()
        if not videos:
            discovery_jobs[job_id]["status"] = "failed"
            discovery_jobs[job_id]["error"] = "No discovered videos to audit"
            return
            
        # Select candidates (e.g. ones not already scored/processed)
        candidates = videos[:limit]
        
        # Sequentially score candidates
        scored_results = []
        for i, video in enumerate(candidates):
            discovery_jobs[job_id]["progress"] = int((i / limit) * 100)
            
            # Run scorer (blocking IO)
            scored = await asyncio.to_thread(video_scorer.score_video, video)
            scored_results.append(scored)
            
        # Update persistent results with scores
        youtube_discovery.save_results(scored_results, overwrite=False)
        
        discovery_jobs[job_id]["status"] = "completed"
        discovery_jobs[job_id]["progress"] = 100
        discovery_jobs[job_id]["results"] = scored_results
        
    except Exception as e:
        logger.error(f"Audit task {job_id} failed: {e}")
        discovery_jobs[job_id]["status"] = "failed"
        discovery_jobs[job_id]["error"] = str(e)
