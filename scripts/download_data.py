#!/usr/bin/env python3
"""
Data download utility for SignVerse system.
Supports YouTube downloads, sample datasets, and external sources.
"""
import argparse
import yt_dlp
import requests
import os
from pathlib import Path
from typing import List, Optional
import json
from loguru import logger
from urllib.parse import urlparse

from configs import get_config
from core.data_manager import data_manager, DataSource

class DataDownloader:
    """Handles data download from various sources."""
    
    def __init__(self):
        self.config = get_config()
    
    def download_youtube_video(self, url: str, output_dir: Optional[str] = None) -> str:
        """Download video from YouTube."""
        output_dir = output_dir or self.config.pipelines.ingestion.youtube.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': f'{output_dir}/youtube_%(id)s_%(title)s.%(ext)s',
            'quiet': False,
            'no_warnings': True,
            'max_filesize': 500000000,  # 500MB limit
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                logger.success(f"Downloaded: {filename}")
                return filename
        except Exception as e:
            logger.error(f"YouTube download failed: {e}")
            raise
    
    def download_sample_dataset(self, dataset_name: str = "signlanguage-sample") -> List[str]:
        """Download sample dataset for testing."""
        sample_datasets = {
            "signlanguage-sample": {
                "urls": [
                    "https://example.com/sample1.mp4",
                    "https://example.com/sample2.mp4"
                ],
                "description": "Sample sign language videos"
            }
        }
        
        if dataset_name not in sample_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(sample_datasets.keys())}")
        
        dataset = sample_datasets[dataset_name]
        downloaded_files = []
        
        for url in dataset["urls"]:
            try:
                filename = self._download_file(url, "uploads")
                downloaded_files.append(filename)
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
        
        return downloaded_files
    
    def _download_file(self, url: str, source: str) -> str:
        """Download a file from URL."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        save_path = data_manager.get_video_path(DataSource(source), filename)
        
        # Download file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded: {save_path}")
        return str(save_path)
    
    def list_downloaded_videos(self, source: Optional[str] = None) -> List[str]:
        """List all downloaded videos."""
        videos = []
        sources = [source] if source else ["uploads", "youtube_videos"]
        
        for src in sources:
            source_dir = data_manager.get_video_path(DataSource(src), "")
            if source_dir.exists():
                videos.extend([str(f) for f in source_dir.glob("*") if f.is_file()])
        
        return videos

def main():
    parser = argparse.ArgumentParser(description="SignVerse Data Download Utility")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # YouTube download
    youtube_parser = subparsers.add_parser('youtube', help='Download YouTube video')
    youtube_parser.add_argument('url', help='YouTube video URL')
    youtube_parser.add_argument('--output-dir', help='Custom output directory')
    
    # Sample dataset
    sample_parser = subparsers.add_parser('sample', help='Download sample dataset')
    sample_parser.add_argument('--dataset', default='signlanguage-sample', 
                              help='Dataset name (default: signlanguage-sample)')
    
    # List videos
    list_parser = subparsers.add_parser('list', help='List downloaded videos')
    list_parser.add_argument('--source', choices=['uploads', 'youtube'], 
                            help='Filter by source')
    
    args = parser.parse_args()
    downloader = DataDownloader()
    
    if args.command == 'youtube':
        filename = downloader.download_youtube_video(args.url, args.output_dir)
        print(f"Downloaded: {filename}")
    
    elif args.command == 'sample':
        files = downloader.download_sample_dataset(args.dataset)
        print(f"Downloaded {len(files)} sample files")
        for f in files:
            print(f"  - {f}")
    
    elif args.command == 'list':
        videos = downloader.list_downloaded_videos(args.source)
        print(f"Found {len(videos)} videos:")
        for video in videos:
            print(f"  - {video}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
