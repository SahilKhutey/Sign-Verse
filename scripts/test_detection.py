import sys
import os
import numpy as np
from loguru import logger

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.perception.detection import PersonDetector

def test_person_detector():
    logger.info("Starting PersonDetector verification...")
    
    try:
        # Initialize detector (using 'n' for faster download in test)
        detector = PersonDetector(model_size="n", confidence_threshold=0.6)
        
        # Create a blank black image (BGR)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection
        logger.info("Running detection on blank frame...")
        detections = detector.detect(frame)
        
        logger.info(f"Detection successful. Found {len(detections)} persons in blank frame.")
        
        # Check output type
        assert isinstance(detections, list), "Detections should be a list"
        
        logger.success("PersonDetector verification PASSED!")
        
    except Exception as e:
        logger.error(f"PersonDetector verification FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_person_detector()
