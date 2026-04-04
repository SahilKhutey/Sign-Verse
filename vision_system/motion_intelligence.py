"""
Motion Intelligence Layer — High Fidelity Sign Feature Engineering
Transforms 543-landmark MediaPipe Holistic landmarks into 848-dim motion intelligence vectors.
"""

import numpy as np
import time
from vision_system.features import hand_distances, body_orientation, mouth_open
from vision_system.holistic_tracker import HolisticTracker

class MotionIntelligence:
    def __init__(self):
        self.tracker = HolisticTracker()
        self.last_features = None # Stores the 424-dim vector from the previous frame

    def extract_1629_dim(self, landmarks_dict):
        """
        Extracts 1,629-dim base geometric vector (543 landmarks * 3).
        """
        # Flattened positions
        pos = []
        for part in ["pose", "face", "left_hand", "right_hand"]:
            data = landmarks_dict.get(part)
            if data is not None:
                pos.append(data.flatten())
            else:
                # Part missing (e.g., hand out of frame) - Pad with zeros
                dim = 33*3 if part == "pose" else (468*3 if part == "face" else 21*3)
                pos.append(np.zeros(dim))
        
        return np.concatenate(pos)

    def process_frame(self, frame):
        """
        Processes a raw BGR frame and returns the 3,258-dim motion intelligence vector.
        [Position(1629), Velocity(1629)]
        """
        results = self.tracker.process(frame)
        numpy_landmarks = self.tracker.get_full_vectors(results)
        
        current_features = self.extract_1629_dim(numpy_landmarks)
        
        # Velocity compute (delta features)
        if self.last_features is None:
            velocity = np.zeros_like(current_features)
        else:
            velocity = current_features - self.last_features
            
        # Update last features
        self.last_features = current_features.copy()
        
        # Concatenate Position + Velocity = 3,258 dimensions
        motion_vector = np.concatenate([current_features, velocity])
        
        return {
            "motion_vector": motion_vector,
            "latency_ms": results["latency_ms"],
            "features_1629": current_features,
            "velocity_1629": velocity
        }

if __name__ == "__main__":
    # Self-test with random frame
    import cv2
    mi = MotionIntelligence()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = mi.process_frame(dummy_frame)
    print(f"Motion intelligence vector shape: {out['motion_vector'].shape}")
    print(f"Latency: {out['latency_ms']}ms")
