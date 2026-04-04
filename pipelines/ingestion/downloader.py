"""
Generic file downloader and upload processor.
"""
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from prefect import task, flow
from typing import List
import time

from configs import get_config
from core.data_manager import data_manager

class UploadHandler(FileSystemEventHandler):
    """Watchdog handler for new file uploads."""
    
    def __init__(self, pipeline_callback):
        self.pipeline_callback = pipeline_callback
    
    def on_created(self, event):
        if not event.is_directory:
            self.pipeline_callback(event.src_path)

@task(name="downloader.process_uploaded_video")
def process_uploaded_video(video_path: str) -> str:
    """
    Process an uploaded video file (validation, moving to appropriate location).
    
    Args:
        video_path: Path to the uploaded video file
        
    Returns:
        Final path to the processed video
    """
    config = get_config()
    path = Path(video_path)
    
    # Validate file extension
    allowed_exts = config.pipelines.ingestion.upload.allowed_extensions
    if path.suffix.lower() not in allowed_exts:
        raise ValueError(f"Invalid file extension: {path.suffix}. Allowed: {allowed_exts}")
    
    # Create destination filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_filename = f"upload_{path.stem}_{timestamp}{path.suffix}"
    dest_path = Path(config.pipelines.ingestion.upload.output_dir) / dest_filename
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move file to permanent location
    shutil.move(str(path), str(dest_path))
    
    return str(dest_path)

@flow(name="ingestion_monitor")
def start_upload_monitor():
    """Start monitoring for new file uploads."""
    config = get_config()
    watch_dir = Path(config.pipelines.ingestion.upload.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    
    def process_new_file(file_path):
        try:
            process_uploaded_video(file_path)
        except Exception as e:
            print(f"Error processing uploaded file {file_path}: {e}")
    
    event_handler = UploadHandler(process_new_file)
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
