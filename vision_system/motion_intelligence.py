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

    def extract_424_dim(self, landmarks_dict):
        """
        Extracts 424-dim base geometric vector.
        lh_dist (210) + rh_dist (210) + body (3) + mouth (1) = 424.
        """
        lh = hand_distances(landmarks_dict.get("left_hand"))
        rh = hand_distances(landmarks_dict.get("right_hand"))
        body = body_orientation(landmarks_dict.get("pose"))
        # Face landmarks for mouth: 13, 14 are indices for upper/lower inner lip
        # In MediaPipe Face Mesh, 13, 14 are often used.
        mouth = np.array([mouth_open(landmarks_dict.get("face"))])

        return np.concatenate([lh, rh, body, mouth])

    def process_frame(self, frame):
        """
        Processes a raw BGR frame and returns the 848-dim motion intelligence vector.
        [Position(424), Velocity(424)]
        """
        results = self.tracker.process(frame)
        numpy_landmarks = self.tracker.get_full_vectors(results)
        
        current_features = self.extract_424_dim(numpy_landmarks)
        
        # Velocity compute (delta features)
        if self.last_features is None:
            velocity = np.zeros_like(current_features)
        else:
            velocity = current_features - self.last_features
            
        # Update last features
        self.last_features = current_features.copy()
        
        # Concatenate Position + Velocity = 848 dimensions
        motion_vector = np.concatenate([current_features, velocity])
        
        return {
            "motion_vector": motion_vector,
            "latency_ms": results["latency_ms"],
            "features_424": current_features,
            "velocity_424": velocity
        }

if __name__ == "__main__":
    # Self-test with random frame
    import cv2
    mi = MotionIntelligence()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = mi.process_frame(dummy_frame)
    print(f"Motion intelligence vector shape: {out['motion_vector'].shape}")
    print(f"Latency: {out['latency_ms']}ms")
