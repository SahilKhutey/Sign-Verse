import sys
import os

# Add the motion_capture directory to path if needed (though it should be fine if in signverse-ai root)
from motion_capture.pose_extractor import NeuralMotionCapture

class FullBodyTracker:
    """
    Wrapper for NeuralMotionCapture to provide the interface expected by 
    the dataset collector and real-time mapping scripts.
    """
    def __init__(self):
        self.detector = NeuralMotionCapture()

    def process(self, frame):
        """
        Processes a frame and returns a flat array of keypoints.
        Shape: (225,) -> 99 (body) + 63 (hand1) + 63 (hand2)
        """
        # NeuralMotionCapture.extract_pose returns a numpy array of keypoints
        return self.detector.extract_pose(frame)

    def visualize(self, frame, draw=True):
        """
        Draws landmarks on the frame.
        """
        return self.detector.visualize(frame, draw=draw)

if __name__ == "__main__":
    # Test the wrapper
    import cv2
    tracker = FullBodyTracker()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        data = tracker.process(frame)
        frame = tracker.visualize(frame)
        
        cv2.imshow("FullBodyTracker Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
