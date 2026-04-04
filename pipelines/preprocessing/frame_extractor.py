"""
Video frame extraction pipeline component.
"""
import cv2
from pathlib import Path
from prefect import task
from loguru import logger
from typing import List, Optional

from configs import get_config

@task(name="frame_extractor.extract_frames")
def extract_frames(video_path: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to input video file
        output_dir: Custom output directory (optional)
        
    Returns:
        List of paths to extracted frames
    """
    config = get_config()
    output_dir = output_dir or config.pipelines.preprocessing.frame_extraction.output_dir
    fps = config.pipelines.preprocessing.frame_extraction.fps
    format = config.pipelines.preprocessing.frame_extraction.format
    quality = config.pipelines.preprocessing.frame_extraction.quality
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_filename = f"{Path(video_path).stem}_frame_{frame_count:06d}.{format}"
            frame_path = Path(output_dir) / frame_filename
            
            # Save frame with specified quality
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(frame_path), frame)
            
            extracted_frames.append(str(frame_path))
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(extracted_frames)} frames from {video_path}")
    return extracted_frames

@task(name="frame_extractor.extract_key_frames")
def extract_key_frames(video_path: str, output_dir: str, method: str = "uniform") -> List[str]:
    """
    Extract key frames using different methods.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for frames
        method: Extraction method ('uniform', 'scene_change')
        
    Returns:
        List of paths to key frames
    """
    if method == "uniform":
        return extract_frames(video_path, output_dir)
    elif method == "scene_change":
        # Implementation for scene change detection
        return extract_scene_change_frames(video_path, output_dir)
    else:
        raise ValueError(f"Unknown frame extraction method: {method}")

def extract_scene_change_frames(video_path: str, output_dir: str) -> List[str]:
    """Extract frames at scene changes using OpenCV."""
    # Implementation for scene change detection
    pass
