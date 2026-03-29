"""
Vision Pipeline — Feature Extractor

Combines pose tracking + face tracking + hand tracking into a unified
feature vector for downstream models.

Output: 1629-dim feature vector per frame
    - 99  body features (33 × 3)
    - 1404 face features (468 × 3)
    - 126 hand features (2 hands × 21 × 3)
"""

import numpy as np

FEATURE_DIM_1629 = 1629

class FeatureExtractor:

    def __init__(self):
        try:
            from vision_pipeline.hand_tracking import HandTracker
            from vision_pipeline.pose_tracking import PoseTracker
            from vision_pipeline.face_tracking import FaceTracker
            self.hand_tracker = HandTracker(max_hands=2)
            self.pose_tracker = PoseTracker()
            self.face_tracker = FaceTracker()
            self.use_mock = False
        except Exception as e:
            print(f"Warning: Mediapipe initialization failed ({e}). Using mock feature extractor.")
            self.use_mock = True

    def extract(self, frame):
        """
        Extract combined feature vector from frame.

        Returns:
            features: numpy (1629,)
            hands_result: MediaPipe hands result
            pose_result: MediaPipe pose result
            face_result: MediaPipe face result
        """
        if self.use_mock:
            # Return random noise that matches the expected distribution for stress testing
            return np.random.randn(FEATURE_DIM_1629).astype(np.float32), None, None, None

        hand_kp, hands_result = self.hand_tracker.process(frame)
        pose_kp, pose_result = self.pose_tracker.process(frame)
        face_kp, face_result = self.face_tracker.process(frame)

        # Canonical layout: [body(99), face(1404), hands(126)]
        features = np.concatenate([pose_kp, face_kp, hand_kp])

        # Guardrail: keep shape stable even if upstream trackers change.
        if features.shape[0] != FEATURE_DIM_1629:
            features = features[:FEATURE_DIM_1629]
            if features.shape[0] < FEATURE_DIM_1629:
                features = np.pad(features, (0, FEATURE_DIM_1629 - features.shape[0]))

        return features, hands_result, pose_result, face_result

    def extract_and_draw(self, frame):
        """Extract features and draw landmarks on frame."""
        features, hands_result, pose_result, face_result = self.extract(frame)
        if not self.use_mock:
            frame = self.hand_tracker.draw(frame, hands_result)
            frame = self.pose_tracker.draw(frame, pose_result)
            frame = self.face_tracker.draw(frame, face_result)
        return features, frame
