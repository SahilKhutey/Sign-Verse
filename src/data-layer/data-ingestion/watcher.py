import time
import os
import sys
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sign-Verse.Watcher")

class VideoIngestionHandler(FileSystemEventHandler):
    """
    Standard Ingestion Handler: Detects new video files (.mp4, .mkv, .avi) 
    and notifies the processing-layer.
    """
    def __init__(self, target_dir):
        self.target_dir = target_dir

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_ext = os.path.splitext(event.src_path)[1].lower()
        if file_ext in ['.mp4', '.mkv', '.avi', '.mov']:
            logger.info(f"New video file detected: {event.src_path}")
            # Placeholder for actual ingestion logic (e.g., POST to pose-service)
            self._notify_processing_layer(event.src_path)

    def _notify_processing_layer(self, file_path):
        """
        Signals the Pose Service to start extraction on the new file.
        """
        logger.info(f"Notifying Processing Layer for: {os.path.basename(file_path)}")
        # Logic to call pose-service API goes here

if __name__ == "__main__":
    watch_path = os.getenv("WATCH_PATH", "/data/ingest")
    if not os.path.exists(watch_path):
        os.makedirs(watch_path)
    
    event_handler = VideoIngestionHandler(watch_path)
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    
    logger.info(f"Starting Sign-Verse Data Ingestion Watcher on {watch_path}...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
