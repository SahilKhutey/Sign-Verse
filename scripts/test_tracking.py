import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import List
from loguru import logger

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.perception.tracking import ByteTracker

@dataclass
class MockDetection:
    bbox: np.ndarray
    confidence: float

def test_byte_tracker():
    logger.info("Starting ByteTracker verification...")
    
    tracker = ByteTracker(max_lost=5, iou_threshold=0.3)
    
    # Frame 1: One detection
    logger.info("Frame 1: Initial detection")
    dets_f1 = [MockDetection(np.array([10, 10, 50, 50]), 0.9)]
    tracks_f1 = tracker.update(dets_f1, 0)
    
    assert len(tracks_f1) == 1
    track_id = tracks_f1[0].track_id
    logger.info(f"Assigned ID: {track_id}")
    
    # Frame 2: Same detection, slightly moved
    logger.info("Frame 2: Object moved")
    dets_f2 = [MockDetection(np.array([12, 12, 52, 52]), 0.85)]
    tracks_f2 = tracker.update(dets_f2, 1)
    
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == track_id
    logger.info(f"Maintained ID: {tracks_f2[0].track_id}")
    
    # Frame 3: Object disappears (lost)
    logger.info("Frame 3: Object lost")
    tracks_f3 = tracker.update([], 2)
    assert len(tracks_f3) == 0
    assert len(tracker.lost_tracks) == 1
    
    # Frame 4: Object reappears
    logger.info("Frame 4: Object reappears")
    dets_f4 = [MockDetection(np.array([13, 13, 53, 53]), 0.88)]
    tracks_f4 = tracker.update(dets_f4, 3)
    
    assert len(tracks_f4) == 1
    assert tracks_f4[0].track_id == track_id
    logger.info(f"Recovered ID: {tracks_f4[0].track_id}")
    
    # Frame 5: New object appears
    logger.info("Frame 5: New object appears")
    dets_f5 = [
        MockDetection(np.array([13, 13, 53, 53]), 0.9),
        MockDetection(np.array([100, 100, 150, 150]), 0.95)
    ]
    tracks_f5 = tracker.update(dets_f5, 4)
    assert len(tracks_f5) == 2
    ids = [t.track_id for t in tracks_f5]
    assert track_id in ids
    new_id = [i for i in ids if i != track_id][0]
    logger.info(f"Tracked IDs: {ids}")
    
    logger.success("ByteTracker verification PASSED!")

if __name__ == "__main__":
    test_byte_tracker()
