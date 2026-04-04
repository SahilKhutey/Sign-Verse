#!/usr/bin/env python3
"""
Pipeline execution script for SignVerse system.
Orchestrates complete data processing pipelines.
"""
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import json
from loguru import logger

from configs import get_config
from pipelines.orchestration.pipeline_runner import run_full_pipeline, run_quick_pose_pipeline
from core.data_manager import data_manager, DataSource

class PipelineRunner:
    """Orchestrates pipeline execution."""
    
    def __init__(self):
        self.config = get_config()
    
    async def run_full_pipeline(self, video_path: str, source: str = "upload") -> Dict[str, Any]:
        """Run the complete processing pipeline."""
        try:
            # Validate video file
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Run pipeline
            result = await run_full_pipeline(str(video_file), source)
            
            logger.success(f"Pipeline completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    async def run_batch_pipeline(self, video_dir: str, source: str = "upload") -> List[Dict[str, Any]]:
        """Run pipeline on multiple videos."""
        video_directory = Path(video_dir)
        if not video_directory.exists():
            raise FileNotFoundError(f"Directory not found: {video_dir}")
        
        video_files = list(video_directory.glob("*.[mp4|avi|mov|mkv]"))
        results = []
        
        for video_file in video_files:
            try:
                result = await self.run_full_pipeline(str(video_file), source)
                results.append({
                    "file": str(video_file),
                    "success": True,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "file": str(video_file),
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def list_available_pipelines(self) -> List[str]:
        """List available pipeline configurations."""
        pipelines = get_config().workflows
        return list(pipelines.keys())

def main():
    parser = argparse.ArgumentParser(description="SignVerse Pipeline Runner")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run pipeline
    run_parser = subparsers.add_parser('run', help='Run processing pipeline')
    run_parser.add_argument('video_path', help='Path to video file')
    run_parser.add_argument('--source', choices=['upload', 'youtube'], default='upload',
                          help='Video source type')
    run_parser.add_argument('--pipeline', default='full_processing_pipeline',
                          help='Pipeline configuration name')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Batch process videos')
    batch_parser.add_argument('directory', help='Directory containing videos')
    batch_parser.add_argument('--source', choices=['upload', 'youtube'], default='upload',
                           help='Video source type')
    
    # List pipelines
    list_parser = subparsers.add_parser('list', help='List available pipelines')
    
    args = parser.parse_args()
    runner = PipelineRunner()
    
    if args.command == 'run':
        asyncio.run(runner.run_full_pipeline(args.video_path, args.source))
    
    elif args.command == 'batch':
        results = asyncio.run(runner.run_batch_pipeline(args.directory, args.source))
        print(f"Processed {len(results)} videos:")
        for result in results:
            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['file']}")
    
    elif args.command == 'list':
        pipelines = runner.list_available_pipelines()
        print("Available pipelines:")
        for pipeline in pipelines:
            print(f"  - {pipeline}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
