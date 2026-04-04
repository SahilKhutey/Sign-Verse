import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from core.settings import settings
from .youtube_discovery import youtube_discovery
from .metadata_filter import metadata_filter
from .video_scorer import video_scorer
from .youtube_pipeline import youtube_pipeline

class DatasetBuilder:
    def __init__(self):
        self.dataset_dir = settings.DATA_DIR / "datasets"
        os.makedirs(self.dataset_dir, exist_ok=True)
        # Concurrency control for simultaneous builds (based on systems/config)
        self._build_semaphore = asyncio.Semaphore(1) 
    
    async def build_dataset(self, 
                            max_videos: int = 20, 
                            job_id: Optional[str] = None,
                            resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Build a dataset by discovering and processing videos.
        Supports resume capability by checking existing processing_results.json.
        """
        async with self._build_semaphore:
            dataset_id = resume_from or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_path = self.dataset_dir / dataset_id
            os.makedirs(dataset_path, exist_ok=True)
            
            logger.info(f"Starting dataset build: {dataset_id} (max {max_videos} videos)")
            
            # 1. Load or Discover candidates
            discovery_file = dataset_path / "discovered_candidates.json"
            if os.path.exists(discovery_file):
                logger.info("Found existing discovery results, resuming...")
                with open(discovery_file, "r") as f:
                    candidates = json.load(f)
            else:
                logger.info("Discovering videos via search...")
                discovered = await asyncio.to_thread(youtube_discovery.discover_videos, max_per_category=5)
                candidates = youtube_discovery.remove_duplicates(discovered)
                with open(discovery_file, "w") as f:
                    json.dump(candidates, f, indent=2)

            # 2. Metadata filtering (always fast)
            filtered = metadata_filter.filter_batch(candidates)
            logger.info(f"Filtered {len(candidates)} -> {len(filtered)} via metadata")
            
            # 3. Content scoring (Audit)
            # Only score what we haven't scored yet if resuming
            scored_file = dataset_path / "scored_videos.json"
            scored_videos = []
            if os.path.exists(scored_file):
                with open(scored_file, "r") as f:
                    scored_videos = json.load(f)
            
            scored_ids = {v['id'] for v in scored_videos}
            remain_to_score = [v for v in filtered if v['id'] not in scored_ids]
            
            logger.info(f"Candidates to score: {len(remain_to_score)}")
            if remain_to_score:
                num_to_score = max_videos - len(scored_videos)
                logger.info(f"Scoring up to {num_to_score} new candidates...")
                for v in remain_to_score[:num_to_score]:
                    scored = await asyncio.to_thread(video_scorer.score_video, v)
                    scored_videos.append(scored)
                    # Incremental save
                    with open(scored_file, "w") as f:
                        json.dump(scored_videos, f, indent=2)

            accepted_videos = [v for v in scored_videos if v.get('status') == 'accepted']
            logger.info(f"Accepted {len(accepted_videos)} videos for processing. Scored total: {len(scored_videos)}")
            
            # 4. Final Processing
            process_file = dataset_path / "processing_results.json"
            processed_results = []
            if os.path.exists(process_file):
                with open(process_file, "r") as f:
                    processed_results = json.load(f)
            
            processed_ids = {v['video']['id'] for v in processed_results}
            to_process = [v for v in accepted_videos if v['id'] not in processed_ids]
            logger.info(f"Remaining to process: {len(to_process)}")
            
            for video in to_process:
                logger.info(f"Starting processing for: {video['title']} ({video['url']})")
                try:
                    result = await youtube_pipeline.run_pipeline(video['url'], video['id'])
                    logger.info(f"Pipeline result for {video['id']}: {result.get('success')}")
                    if result.get('success'):
                        processed_results.append({
                            'video': video,
                            'processing_result': result,
                            'processed_at': datetime.now().isoformat()
                        })
                        # Save state after each success
                        with open(process_file, "w") as f:
                            json.dump(processed_results, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to process {video['id']}: {e}")

            # 5. Summary Generation
            summary = self._generate_summary(dataset_id, candidates, filtered, accepted_videos, processed_results)
            with open(dataset_path / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.success(f"Dataset build complete: {dataset_id}")
            return summary

    def _generate_summary(self, dataset_id, candidates, filtered, accepted, processed):
        """Helper to create dataset metrics."""
        categories = {}
        for v in accepted:
            cat = v.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
        return {
            "dataset_id": dataset_id,
            "created_at": datetime.now().isoformat(),
            "metrics": {
                "discovered": len(candidates),
                "metadata_passed": len(filtered),
                "visual_accepted": len(accepted),
                "pipeline_completed": len(processed)
            },
            "categories": categories,
            "average_quality_score": sum(v.get('score', 0) for v in accepted) / (len(accepted) or 1)
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets and their summaries."""
        datasets = []
        if not os.path.exists(self.dataset_dir):
            return []
            
        for d in os.listdir(self.dataset_dir):
            summary_path = self.dataset_dir / d / "summary.json"
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    datasets.append(json.load(f))
        return datasets

# Global dataset builder instance
dataset_builder = DatasetBuilder()
