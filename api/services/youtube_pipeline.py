"""
YouTube video processing pipeline with yt-dlp integration and perception analysis.
"""
import os
import json
import cv2
import yt_dlp
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from core.settings import settings
from .perception_service import process_frame
from .content_analyzer import content_analyzer

class YouTubePipeline:
    def __init__(self):
        # Configuration
        self.base_dir = settings.DATA_DIR
        self.youtube_data_dir = settings.youtube.base_path
        self.max_concurrent = settings.youtube.max_concurrent_jobs
        self.cookies_path = settings.youtube.cookies_path
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.youtube_data_dir / "raw", exist_ok=True)
        os.makedirs(self.youtube_data_dir / "processed", exist_ok=True)
    
    def download_video(self, url: str) -> Dict[str, Any]:
        """Download YouTube video using yt-dlp with optional cookie support."""
        output_path = self.youtube_data_dir / "raw"
        os.makedirs(output_path, exist_ok=True)

        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': f'{output_path}/%(id)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'ignoreerrors': False,
        }

        # Add cookie support for private/restricted videos
        if self.cookies_path and os.path.exists(self.cookies_path):
            logger.info(f"Using YouTube cookies from {self.cookies_path}")
            ydl_opts['cookiefile'] = str(self.cookies_path)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                file_ext = info.get('ext', 'mp4')
                filename = os.path.join(output_path, f"{video_id}.{file_ext}")

            return {
                "video_id": video_id,
                "title": info.get("title", "Unknown"),
                "path": filename,
                "duration": info.get("duration", 0),
                "success": True
            }
        except Exception as e:
            logger.error(f"yt-dlp download failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_frames(self, video_path: str, output_dir: str, fps: int = 5) -> int:
        """Extract frames from video at targeted FPS for efficiency."""
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0: video_fps = 30 # Fallback
        
        interval = max(1, int(video_fps / fps))
        frame_id = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % interval == 0:
                filename = os.path.join(output_dir, f"frame_{saved:06d}.jpg")
                cv2.imwrite(filename, frame)
                saved += 1

            frame_id += 1

        cap.release()
        return saved
    
    def process_frames(self, frames_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Process extracted frames through the perception system."""
        os.makedirs(output_dir, exist_ok=True)
        
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        results = []

        for i, file in enumerate(frame_files):
            path = os.path.join(frames_dir, file)
            frame = cv2.imread(path)
            
            if frame is not None:
                # Call our perception service singleton wrapper
                result = process_frame(frame)
                result["frame_id"] = i
                result["frame_file"] = file
                results.append(result)

                # Save detailed per-frame results
                with open(os.path.join(output_dir, f"frame_{i:06d}.json"), "w") as f:
                    json.dump(result, f, indent=2)

        return results
    
    def generate_events(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate events based on perception analytics."""
        events = []

        for result in results:
            # Map detected persons to generic events
            for person in result.get("persons", []):
                events.append({
                    "event_type": "person_detected",
                    "person_id": person.get("id", "unknown"),
                    "frame_id": result["frame_id"],
                    "timestamp": result.get("timestamp", 0),
                    "position": person.get("position", {}),
                    "confidence": person.get("confidence", 0)
                })
            
            # Map interactions/features to interaction events
            for interaction in result.get("interactions", []):
                events.append({
                    "event_type": interaction.get("type", "interaction"),
                    "person_id": interaction.get("person_id", "unknown"),
                    "object_id": interaction.get("object_id", "unknown"),
                    "frame_id": result["frame_id"],
                    "timestamp": result.get("timestamp", 0),
                    "confidence": interaction.get("confidence", 0)
                })

        return events
    
    def infer_intentions(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer high-level intentions from low-level events."""
        intentions = []
        
        # Person-centric event grouping
        person_events = {}
        for event in events:
            p_id = event.get("person_id")
            if p_id not in person_events:
                person_events[p_id] = []
            person_events[p_id].append(event)
        
        # Analyze temporal patterns for each person
        for p_id, p_events in person_events.items():
            interactions = [e for e in p_events if "interaction" in e["event_type"] or "gesture" in e["event_type"]]
            
            if interactions:
                # Simple logic for 'active state'
                intentions.append({
                    "person_id": p_id,
                    "intention": "human_system_interaction",
                    "confidence": 0.85,
                    "frame_id": interactions[0]["frame_id"],
                    "timestamp": interactions[0]["timestamp"],
                    "signals": list(set(e["event_type"] for e in interactions))
                })
        
        return intentions

    def verify_video_content(self, video_path: str) -> Dict[str, Any]:
        """Perform visual audit of video content before full processing."""
        logger.info(f"Verifying visual content for {video_path}")
        
        frames = content_analyzer.sample_frames(video_path)
        if not frames:
            return {"success": False, "error": "Failed to sample frames"}
            
        human_presence = content_analyzer.detect_humans(frames)
        trackability = content_analyzer.assess_trackability(frames)
        heuristics = content_analyzer.classify_heuristics(frames)
        motion_score = content_analyzer.compute_motion_score(frames)
        
        # Thresholds
        is_valid = human_presence > 0.2 and trackability['total'] > 0.3
        
        return {
            "success": is_valid,
            "human_presence": human_presence,
            "trackability": trackability,
            "heuristics": heuristics,
            "motion_score": motion_score,
            "error": None if is_valid else "Insufficient trackable human motion"
        }

    def cleanup_raw_video(self, video_path: str):
        """Delete raw video file to save space."""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up raw video: {video_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup video {video_path}: {e}")
    
    async def run_pipeline(self, youtube_url: str, job_id: str) -> Dict[str, Any]:
        """Orchestrate the entire pipeline with concurrency limiting."""
        async with self._semaphore:
            logger.info(f"Job {job_id} starting processing for {youtube_url}")
            
            try:
                # Step 1: Download
                # yt-dlp is blocking, but we'll run it in a thread if needed
                # For now, simplistic approach
                download_result = await asyncio.to_thread(self.download_video, youtube_url)
                if not download_result["success"]:
                    return {"success": False, "error": download_result["error"]}
                
                # Step 2: Visual Verification
                verify_result = await asyncio.to_thread(self.verify_video_content, download_result["path"])
                if not verify_result["success"]:
                    logger.warning(f"Video {download_result['video_id']} failed visual verification: {verify_result['error']}")
                    self.cleanup_raw_video(download_result["path"])
                    return {"success": False, "error": verify_result["error"], "verification_details": verify_result}

                job_dir = self.youtube_data_dir / "processed" / job_id
                frames_dir = job_dir / "frames"
                results_dir = job_dir / "results"
                os.makedirs(job_dir, exist_ok=True)
                
                # Step 3: Extract Frames
                frame_count = await asyncio.to_thread(
                    self.extract_frames, download_result["path"], str(frames_dir)
                )
                
                # Step 3: Perception Analysis
                results = await asyncio.to_thread(
                    self.process_frames, str(frames_dir), str(results_dir)
                )
                
                # Step 4: Knowledge Refinement
                events = self.generate_events(results)
                intentions = self.infer_intentions(events)
                
                # Persistence
                summary = {
                    "job_id": job_id,
                    "video_id": download_result["video_id"],
                    "title": download_result["title"],
                    "duration": download_result["duration"],
                    "frames_processed": len(results),
                    "persons_detected": len(set(e["person_id"] for e in events if e["event_type"] == "person_detected")),
                    "events_count": len(events),
                    "intentions_count": len(intentions),
                    "success": True
                }
                
                await asyncio.to_thread(self._save_summary, job_dir, summary, events, intentions)
                logger.success(f"Job {job_id} completed successfully.")
                return summary
                
            except Exception as e:
                logger.error(f"Pipeline error for job {job_id}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return {"success": False, "error": str(e)}

    def _save_summary(self, job_dir, summary, events, intentions):
        """Helper to save JSON artifacts to the filesystem."""
        with open(job_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(job_dir / "events.json", "w") as f:
            json.dump(events, f, indent=2)
        with open(job_dir / "intentions.json", "w") as f:
            json.dump(intentions, f, indent=2)

# Global singleton instance
youtube_pipeline = YouTubePipeline()
