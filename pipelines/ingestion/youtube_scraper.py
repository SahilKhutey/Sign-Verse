"""
YouTube video ingestion pipeline component.
"""
import yt_dlp
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from prefect import task
from loguru import logger

from configs import get_config
from core.data_models import DataSource

@task(name="youtube_scraper.search_videos")
def search_youtube_videos(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of video metadata dictionaries
    """
    config = get_config()
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_json': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            videos = result.get('entries', [])
            
            logger.info(f"Found {len(videos)} videos for query: '{query}'")
            return videos
            
    except Exception as e:
        logger.error(f"YouTube search failed: {e}")
        raise

@task(name="youtube_scraper.download_video")
def download_youtube_video(video_url: str, output_dir: Optional[str] = None) -> str:
    """
    Download a YouTube video with specified quality.
    
    Args:
        video_url: YouTube video URL
        output_dir: Custom output directory (optional)
        
    Returns:
        Path to the downloaded video file
    """
    config = get_config()
    output_dir = output_dir or config.pipelines.ingestion.youtube.output_dir
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_template = f"{output_dir}/youtube_%(id)s_{timestamp}.%(ext)s"
    
    ydl_opts = {
        'format': 'best[height<=720]',  # Max 720p
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': True,
        'max_filesize': 500000000,  # 500MB limit
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            filename = ydl.prepare_filename(info)
            
            logger.success(f"Downloaded video: {filename}")
            return filename
            
    except Exception as e:
        logger.error(f"YouTube download failed: {e}")
        raise
