import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pre-mock to avoid real vision/video initialization
mock_perception = MagicMock()
sys.modules['api.services.perception_service'] = mock_perception
mock_pipeline = MagicMock()
sys.modules['api.services.youtube_pipeline'] = mock_pipeline
mock_analyzer = MagicMock()
sys.modules['api.services.content_analyzer'] = mock_analyzer

from api.services.video_scorer import VideoScorer

def test_video_scorer():
    scorer = VideoScorer()
    mock_video = {
        "id": "test_vid",
        "url": "https://youtube.com/test",
        "title": "Workout Tutorial"
    }
    
    # 1. Mock Download
    mock_pipeline.youtube_pipeline.download_video.return_value = {
        "success": True, "path": "test_path.mp4"
    }
    
    # 2. Mock Analysis
    # Let's simulate a high-quality video
    mock_analyzer.content_analyzer.sample_frames.return_value = [object()] * 30
    mock_analyzer.content_analyzer.detect_humans.return_value = 1.0 # 100% human presence
    mock_analyzer.content_analyzer.compute_motion_score.return_value = 8.0 # High motion
    mock_analyzer.content_analyzer.assess_trackability.return_value = {"total": 0.9} # High trackability
    mock_analyzer.content_analyzer.classify_heuristics.return_value = {"workout": 1.0} # Relevant
    
    # Run scoring
    with patch('os.path.exists', return_value=True), patch('os.remove'):
        result = scorer.score_video(mock_video)
        
    print(f"Final Score: {result['score']}")
    print(f"Status: {result['status']}")
    
    assert result['score'] > 0.8
    assert result['status'] == 'accepted'
    print("[PASSED] high quality scoring")
    
    # 3. Mock Low Quality
    mock_analyzer.content_analyzer.detect_humans.return_value = 0.1 # No humans
    mock_analyzer.content_analyzer.compute_motion_score.return_value = 0.5 # Low motion
    
    with patch('os.path.exists', return_value=True), patch('os.remove'):
        result = scorer.score_video(mock_video)
        
    print(f"Final Score: {result['score']}")
    assert result['status'] == 'rejected'
    print("[PASSED] low quality rejection")

if __name__ == "__main__":
    try:
        test_video_scorer()
        print("\nAll VideoScorer tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
