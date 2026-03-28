"""
Full-Body Pose Extractor for Sign Language Foundation Model

Captures complete signing information:
  - Hand pose:  21 × 3 × 2 hands = 126 features
  - Body pose:  33 × 3 = 99 features
  - Face mesh:  468 × 3 = 1404 features
  Total: ~1629 features per frame

Uses MediaPipe Holistic for unified extraction.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json


class PoseExtractor:

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_frame(self, frame):
        """
        Extract full pose from a single frame.

        Returns dict with hand, body, and face keypoints.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        pose_data = {
            "left_hand": self._landmarks_to_array(results.left_hand_landmarks, 21),
            "right_hand": self._landmarks_to_array(results.right_hand_landmarks, 21),
            "body": self._landmarks_to_array(results.pose_landmarks, 33),
            "face": self._landmarks_to_array(results.face_landmarks, 468),
        }

        return pose_data

    def extract_flat_vector(self, frame):
        """Extract pose as a single flat feature vector (~1629 values)."""
        pose = self.extract_frame(frame)
        vectors = []
        for key in ["left_hand", "right_hand", "body", "face"]:
            if pose[key] is not None:
                vectors.append(pose[key].flatten())
            else:
                sizes = {"left_hand": 63, "right_hand": 63, "body": 99, "face": 1404}
                vectors.append(np.zeros(sizes[key]))
        return np.concatenate(vectors)

    def extract_video(self, video_path, output_path=None):
        """
        Extract pose keypoints from all frames of a video.

        Returns list of pose dicts, optionally saves to JSON.
        """
        cap = cv2.VideoCapture(video_path)
        all_poses = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pose = self.extract_frame(frame)

            serializable = {}
            for key, val in pose.items():
                serializable[key] = val.tolist() if val is not None else None

            all_poses.append({"frame": frame_id, "pose": serializable})
            frame_id += 1

        cap.release()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_poses, f)

        return all_poses

    def _landmarks_to_array(self, landmarks, expected_count):
        """Convert MediaPipe landmarks to numpy array."""
        if landmarks is None:
            return None

        points = []
        for lm in landmarks.landmark[:expected_count]:
            points.extend([lm.x, lm.y, lm.z])

        return np.array(points)
