import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pre-mock perception_service to avoid MediaPipe initialization in test environment
mock_perception_service = MagicMock()
sys.modules['api.services.perception_service'] = mock_perception_service

from api.services.content_analyzer import ContentAnalyzer

def test_content_analyzer():
    analyzer = ContentAnalyzer()
    
    # Mock frame (100x100 RGB)
    mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [mock_frame] * 5
    
    # Mock perception_service result
    mock_perception_result = {
        "persons": [{
            "id": "person_1",
            "confidence": 0.9,
            "keypoints": [
                {"name": "left_shoulder", "x": 50, "y": 20, "score": 0.9},
                {"name": "right_shoulder", "x": 60, "y": 20, "score": 0.9},
                {"name": "left_hip", "x": 50, "y": 80, "score": 0.9},
                {"name": "right_hip", "x": 60, "y": 80, "score": 0.9},
                {"name": "left_elbow", "x": 40, "y": 10, "score": 0.9} # Raised arm for dance heuristic
            ]
        }],
        "interactions": [{
            "type": "manipulating_tool",
            "confidence": 0.8
        }],
        "objects": [{"type": "dumbbell"}]
    }
    
    # Inject mock result into the pre-mocked service
    mock_perception_service.perception_service.process_frame.return_value = mock_perception_result
    # Also mock the top-level process_frame function if used
    mock_perception_service.process_frame.return_value = mock_perception_result
    
    # 1. Test human detection
    presence = analyzer.detect_humans(frames)
    assert presence == 1.0
    print("[PASSED] detect_humans")
    
    # 2. Test trackability assessment
    trackability = analyzer.assess_trackability(frames)
    assert trackability['total'] > 0.5
    assert trackability['interaction'] > 0.0
    print(f"[PASSED] assess_trackability (total: {trackability['total']:.2f})")
    
    # 3. Test heuristics
    heuristics = analyzer.classify_heuristics(frames)
    assert heuristics['workout'] > 0 or heuristics['dance'] > 0
    print(f"[PASSED] classify_heuristics: {heuristics}")

if __name__ == "__main__":
    try:
        test_content_analyzer()
        print("\nAll ContentAnalyzer tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
