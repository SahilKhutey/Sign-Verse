import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.services.youtube_discovery import YouTubeDiscovery

def test_discovery():
    discovery = YouTubeDiscovery()
    
    # Mock yt-dlp response
    mock_entry = {
        "id": "test_id_1",
        "title": "Test Video 1",
        "url": "https://www.youtube.com/watch?v=test_id_1",
        "duration": 100,
        "view_count": 5000,
        "upload_date": "20230101",
        "thumbnail": "http://example.com/thumb.jpg",
        "channel": "Test Channel"
    }
    
    mock_results = {
        "entries": [mock_entry]
    }
    
    with patch('yt_dlp.YoutubeDL') as mock_ydl:
        instance = mock_ydl.return_value.__enter__.return_value
        instance.extract_info.return_value = mock_results
        
        # Test search_videos
        videos = discovery.search_videos("test query", max_results=1)
        assert len(videos) == 1
        assert videos[0]['id'] == "test_id_1"
        print("✓ search_videos passed")
        
        # Test discover_videos with filters
        # Mock categories to be small for test
        discovery.categories = ["dance"]
        
        # 1. Positive required (should fail now with 'Test Video 1')
        mock_entry['title'] = "Random Video"
        videos = discovery.discover_videos(max_per_category=1)
        assert len(videos) == 0
        print("✓ positive required filter passed")

        # 2. Negative title filter
        mock_entry['title'] = "Dance Competition Compilation 2023" # Has 'compilation' (negative)
        videos = discovery.discover_videos(max_per_category=1)
        assert len(videos) == 0
        print("✓ negative keyword filter passed")

        # 3. Successful selection
        mock_entry['channel'] = "Dance Masterclass"
        mock_entry['title'] = "Ballet Practice Session"
        videos = discovery.discover_videos(max_per_category=1)
        assert len(videos) == 1
        print("✓ positive selection passed")

        # 4. Range and view filters
        videos = discovery.discover_videos(max_per_category=1, filters={"min_views": 10000})
        assert len(videos) == 0
        print("✓ view count filter passed")

if __name__ == "__main__":
    try:
        test_discovery()
        print("\nAll discovery tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
