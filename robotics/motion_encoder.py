import numpy as np
from enum import Enum

class ConfigProfile(Enum):
    FULL = "full"           # All 543 landmarks (1629 values)
    ROBOTICS = "robotics"   # Upper Body + Hands
    SIGN_LANGUAGE = "sign"   # Hands + Face (Eyes/Mouth) + Upper Body
    BASICS = "basics"       # Hands + Key Pose Joints

class MotionEncoder:
    """
    Advanced Motion Encoder with Support for MediaPipe Holistic (543 Landmarks).
    Includes feature engineering for Distances, Angles, and Velocities.
    """

    def __init__(self, profile=ConfigProfile.FULL):
        self.profile = profile
        self.prev_pos = None
        
        # --- Landmark Index Definitions (MediaPipe Holistic) ---
        self.INDEX_BODY = list(range(0, 33))
        self.INDEX_FACE = list(range(33, 501))
        self.INDEX_L_HAND = list(range(501, 522))
        self.INDEX_R_HAND = list(range(522, 543))

        # Specialized Face ROI (offsets added to 33)
        self.MOUTH = [33 + i for i in [0, 13, 14, 17, 37, 39, 40, 61, 78, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 273, 287, 308, 310, 311, 312, 314, 317, 318, 321, 324, 325, 375, 402, 405, 409, 415]]
        self.EYEBROWS = [33 + i for i in [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]]
        self.EYES = [33 + i for i in [33, 133, 157, 158, 159, 160, 161, 246, 362, 263, 384, 385, 386, 387, 388, 466]]

    def encode_frame(self, raw_landmarks):
        """
        Processes raw landmarks based on the selected profile and extracts features.
        """
        landmarks = raw_landmarks.reshape(-1, 3) 
        
        # 1. Apply Profile Slicing
        selected_landmarks = self._apply_profile(landmarks)
        pos = selected_landmarks.flatten()

        # 2. Advanced Feature Extraction (Pro Tip: Distances + Angles + Velocity)
        features = {
            "positions": pos,
            "velocities": self._extract_velocities(pos),
            "distances": self._extract_distances(landmarks),
            "vector_angles": self._extract_vector_angles(landmarks),
        }

        return features

    def _apply_profile(self, landmarks):
        """Slices the landmarks based on configuration requirements."""
        if self.profile == ConfigProfile.FULL:
            return landmarks
        
        indices = []
        if self.profile == ConfigProfile.ROBOTICS:
            # Upper Body (Shoulders to Wrists) + Both Hands
            indices = [11, 12, 13, 14, 15, 16] + self.INDEX_L_HAND + self.INDEX_R_HAND
        
        elif self.profile == ConfigProfile.SIGN_LANGUAGE:
            # Hands + Face (Mouth/Eyebrows) + Upper Body
            indices = self.MOUTH + self.EYEBROWS + [11, 12, 13, 14] + self.INDEX_L_HAND + self.INDEX_R_HAND
            
        elif self.profile == ConfigProfile.BASICS:
            # Minimal: Hands + Shoulders
            indices = [11, 12] + self.INDEX_L_HAND + self.INDEX_R_HAND
        
        return landmarks[indices]

    def _extract_velocities(self, current_pos):
        if self.prev_pos is None:
            vel = np.zeros_like(current_pos)
        else:
            vel = current_pos - self.prev_pos
        self.prev_pos = current_pos
        return vel

    def _extract_distances(self, landmarks):
        """Calculates Euclidean distances between critical joint pairs."""
        pairs = [
            (11, 13), (13, 15), # Left Arm segments
            (12, 14), (14, 16), # Right Arm segments
            (501, 505), (501, 509), # Left Hand: Wrist to Thumb/Index tip
            (522, 526), (522, 530), # Right Hand: Wrist to Thumb/Index tip
            (505, 526), # Distance between hands (Clapping/Interacting)
            (509, 33 + 13), # Left Index to Mouth
            (530, 33 + 13)  # Right Index to Mouth
        ]
        distances = []
        for p1, p2 in pairs:
            d = np.linalg.norm(landmarks[p1] - landmarks[p2])
            distances.append(d)
        return np.array(distances)

    def _extract_vector_angles(self, landmarks):
        """Calculates 3D vector angles between bone segments."""
        def get_angle(p1, p2, p3):
            # Angle at p2 between p1-p2 and p3-p2
            v1 = landmarks[p1] - landmarks[p2]
            v2 = landmarks[p3] - landmarks[p2]
            unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
            unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
            return np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0))

        angles = [
            get_angle(11, 13, 15), # Left Elbow
            get_angle(12, 14, 16), # Right Elbow
            get_angle(13, 11, 12), # Left Shoulder
            get_angle(14, 12, 11)  # Right Shoulder
        ]
        return np.array(angles)

    def reset(self):
        self.prev_pos = None

if __name__ == "__main__":
    encoder = MotionEncoder(profile=ConfigProfile.ROBOTICS)
    dummy_raw = np.random.rand(1629)
    
    encoded = encoder.encode_frame(dummy_raw)
    print(f"Profile: {encoder.profile.value}")
    print(f"Positions Shape: {encoded['positions'].shape}")
    print(f"Distances: {encoded['distances']}")
    print(f"Vector Angles: {encoded['vector_angles']}")
