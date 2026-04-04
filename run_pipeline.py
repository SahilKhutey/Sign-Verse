#!/usr/bin/env python3
"""
Main pipeline execution entry point.
"""
import argparse
from pathlib import Path
from loguru import logger

from pipelines.orchestration.pipeline_runner import run_full_pipeline, run_quick_pose_pipeline
from pipelines.ingestion.downloader import start_upload_monitor
from pipelines.orchestration.scheduler import PipelineScheduler

def main():
    parser = argparse.ArgumentParser(description="SignVerse Pipeline Runner")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run pipeline command
    pipeline_parser = subparsers.add_parser('run', help='Run a processing pipeline')
    pipeline_parser.add_argument('--pipeline', choices=['full', 'quick'], default='quick',
                               help='Pipeline type to run')
    pipeline_parser.add_argument('--video', required=True, help='Path to video file')
    pipeline_parser.add_argument('--source', choices=['upload', 'youtube'], default='upload',
                               help='Video source type')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor for new uploads')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Start scheduled processing')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return
        
        if args.pipeline == 'full':
            result = run_full_pipeline(str(video_path), args.source)
        else:
            result = run_quick_pose_pipeline(str(video_path))
        
        logger.info(f"Pipeline result: {result}")
    
    elif args.command == 'monitor':
        logger.info("Starting upload monitor...")
        start_upload_monitor()
    
    elif args.command == 'schedule':
        logger.info("Starting pipeline scheduler...")
        scheduler = PipelineScheduler()
        # Note: In a real production setup, the scheduler would be configured
        # dynamically, but here we provide a default daily processing job.
        scheduler.schedule_daily_processing(run_full_pipeline)
        scheduler.start_scheduler()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
