import sys
import os
import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import uuid

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pre-mock perception to avoid MediaPipe dependency errors
mock_perception = MagicMock()
sys.modules['api.services.perception_service'] = mock_perception

from api.services.dataset_builder import dataset_builder

async def test_dataset_builder():
    # 1. Setup Mocking
    with patch('api.services.dataset_builder.youtube_discovery') as mock_discovery, \
         patch('api.services.dataset_builder.metadata_filter') as mock_filter, \
         patch('api.services.dataset_builder.video_scorer') as mock_scorer, \
         patch('api.services.dataset_builder.youtube_pipeline') as mock_pipeline:
         
        # Discovery mock
        mock_discovery.discover_videos.return_value = [{"id": "v1", "title": "Vid 1", "url": "url1"}]
        mock_discovery.remove_duplicates.return_value = [{"id": "v1", "title": "Vid 1", "url": "url1"}]
        
        # Filter mock
        mock_filter.filter_batch.return_value = [{"id": "v1", "title": "Vid 1", "url": "url1"}]
        
        # Scorer mock (synchronous method)
        mock_scorer.score_video = MagicMock(return_value={"id": "v1", "title": "Vid 1", "url": "url1", "status": "accepted", "score": 0.8})
        
        # Pipeline mock (asynchronous method)
        mock_pipeline.run_pipeline = AsyncMock(return_value={"success": True, "job_id": "job1"})
        
        # 2. Execute Build with unique name to avoid resume interference
        test_dataset_name = f"test_dataset_{uuid.uuid4().hex[:8]}"
        print(f"Starting mock dataset build: {test_dataset_name}...")
        summary = await dataset_builder.build_dataset(max_videos=1, resume_from=test_dataset_name)
        
        # 3. Verifications
        print(f"Metrics: {summary['metrics']}")
        assert summary['metrics']['visual_accepted'] == 1
        assert summary['metrics']['pipeline_completed'] == 1
        print("[PASSED] dataset orchestration")
        
        # Verify persistence (summary.json created)
        dataset_path = os.path.join("data", "datasets", test_dataset_name)
        assert os.path.exists(dataset_path)
        assert os.path.exists(os.path.join(dataset_path, "summary.json"))
        print(f"[PASSED] persistence check for {test_dataset_name}")

if __name__ == "__main__":
    try:
        # Run async test
        asyncio.run(test_dataset_builder())
        print("\nAll DatasetBuilder tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
