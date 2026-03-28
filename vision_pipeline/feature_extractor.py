"""
Vision Pipeline — Feature Extractor

Combines hand tracking + pose tracking into a unified
feature vector for downstream models.

Output: 225-dim feature vector per frame
    - 126 hand features (2 hands × 21 × 3)
    - 99  body features (33 × 3)
"""

import numpy as np
from common.keypoint_schema import FEATURE_DIM_225


class FeatureExtractor:

    def __init__(self):
        try:
            from vision_pipeline.hand_tracking import HandTracker
            from vision_pipeline.pose_tracking import PoseTracker
            self.hand_tracker = HandTracker(max_hands=2)
            self.pose_tracker = PoseTracker()
            self.use_mock = False
        except Exception as e:
            print(f"Warning: Mediapipe initialization failed ({e}). Using mock feature extractor.")
            self.use_mock = True

    def extract(self, frame):
        """
        Extract combined feature vector from frame.

        Returns:
            features: numpy (225,)
            hands_result: MediaPipe hands result
            pose_result: MediaPipe pose result
        """
        if self.use_mock:
            # Return random noise that matches the expected distribution for stress testing
            return np.random.randn(FEATURE_DIM_225).astype(np.float32), None, None

        hand_kp, hands_result = self.hand_tracker.process(frame)
        pose_kp, pose_result = self.pose_tracker.process(frame)

        # Canonical layout: [body(99), hands(126)]
        features = np.concatenate([pose_kp, hand_kp])  # 99 + 126 = 225

        # Guardrail: keep shape stable even if upstream trackers change.
        if features.shape[0] != FEATURE_DIM_225:
            features = features[:FEATURE_DIM_225]
            if features.shape[0] < FEATURE_DIM_225:
                features = np.pad(features, (0, FEATURE_DIM_225 - features.shape[0]))

        return features, hands_result, pose_result

    def extract_and_draw(self, frame):
        """Extract features and draw landmarks on frame."""
        features, hands_result, pose_result = self.extract(frame)
        if not self.use_mock:
            frame = self.hand_tracker.draw(frame, hands_result)
            frame = self.pose_tracker.draw(frame, pose_result)
        return features, frame
