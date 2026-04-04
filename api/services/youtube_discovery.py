import yt_dlp
import re
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from core.settings import settings
from .metadata_filter import metadata_filter

class YouTubeDiscovery:
    def __init__(self):
        self.categories = [
            "gym workout", "yoga practice", "calisthenics", "workout routine",
            "dance performance", "dance tutorial", "ballet practice",
            "construction work", "carpentry", "woodworking",
            "painting tutorial", "art demonstration", "drawing tutorial",
            "sports training", "football training", "basketball practice",
            "martial arts", "boxing training", "mma workout",
            "cooking tutorial", "food preparation", "culinary skills",
            "writing practice", "calligraphy", "handwriting tutorial",
            "blacksmith work", "metalworking", "forging tutorial"
        ]
        self.metadata_path = settings.DATA_DIR / "metadata" / "youtube_discovery.json"
        os.makedirs(self.metadata_path.parent, exist_ok=True)
    
    def search_videos(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search YouTube videos using yt-dlp"""
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "force_json": True,
            "no_warnings": True,
            "default_search": f"ytsearch{max_results}:{query}"
        }

        videos = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(query, download=False)
                
                if results and 'entries' in results:
                    for entry in results['entries']:
                        if not entry:
                            continue
                            
                        video_data = {
                            "id": entry.get("id"),
                            "title": entry.get("title", "Unknown"),
                            "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
                            "duration": entry.get("duration", 0),
                            "view_count": entry.get("view_count", 0),
                            "upload_date": entry.get("upload_date", ""),
                            "thumbnail": entry.get("thumbnail"),
                            "channel": entry.get("channel", "Unknown"),
                            "category": query,
                            "discovery_date": datetime.now().isoformat()
                        }
                        videos.append(video_data)
                        
        except Exception as e:
            logger.error(f"Error searching for '{query}': {str(e)}")
        
        return videos
    
    def discover_videos(self, max_per_category: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Discover videos across all categories with optional filtering."""
        all_videos = []
        filters = filters or {}
        
        min_duration = filters.get("min_duration", 30)
        max_duration = filters.get("max_duration", 600)  # Default 10 mins
        min_views = filters.get("min_views", 1000)
        
        for category in self.categories:
            logger.info(f"Searching for '{category}' videos...")
            videos = self.search_videos(category, max_per_category)
            
            # Apply selection filters including advanced metadata filtering
            filtered = []
            for v in videos:
                # 1. Advanced metadata rules
                if not metadata_filter.filter_video(v):
                    continue
                
                # 2. Strict overrides from discovery arguments (if any)
                if v['duration'] < min_duration or v['duration'] > max_duration:
                    continue
                if v['view_count'] < min_views:
                    continue
                
                filtered.append(v)
            
            logger.info(f"Found {len(videos)} - Selected {len(filtered)} after filtering.")
            all_videos.extend(filtered)
            
            # Avoid rate limiting
            import time
            time.sleep(0.5)
        
        unique_videos = self.remove_duplicates(all_videos)
        self.save_results(unique_videos)
        
        return unique_videos
    
    def remove_duplicates(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate videos by ID"""
        seen_ids = set()
        unique_videos = []
        
        for video in videos:
            if video['id'] not in seen_ids:
                seen_ids.add(video['id'])
                unique_videos.append(video)
        
        return unique_videos

    def save_results(self, videos: List[Dict[str, Any]]):
        """Save discovered metadata for long-term persistence."""
        existing_data = []
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception:
                pass
        
        # Merge unique new discoveries
        existing_ids = {v['id'] for v in existing_data}
        new_videos = [v for v in videos if v['id'] not in existing_ids]
        
        updated_data = existing_data + new_videos
        
        with open(self.metadata_path, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        logger.info(f"Persisted {len(new_videos)} new discoveries to {self.metadata_path}")

    def load_cached_results(self) -> List[Dict[str, Any]]:
        """Load previously discovered results."""
        if not os.path.exists(self.metadata_path):
            return []
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

# Global discovery instance
youtube_discovery = YouTubeDiscovery()
