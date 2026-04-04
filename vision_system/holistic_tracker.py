"""
Holistic Tracker — High Fidelity Pose + Hand + Face Tracking
Powered by MediaPipe Holistic (543 landmarks).
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class HolisticTracker:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.last_process_time = 0

    def process(self, frame):
        """
        Processes a single BGR frame.
        Returns a dictionary of landmarks or None if tracking fails.
        """
        start = time.time()
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        self.last_process_time = time.time() - start

        return {
            "pose": results.pose_landmarks,
            "face": results.face_landmarks,
            "left_hand": results.left_hand_landmarks,
            "right_hand": results.right_hand_landmarks,
            "latency_ms": int(self.last_process_time * 1000)
        }

    def landmarks_to_numpy(self, landmarks, count):
        """Converts MediaPipe landmarks to (count, 3) numpy array."""
        if landmarks is None:
            return np.zeros((count, 3))
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    def get_full_vectors(self, results):
        """Extracts all landmarks into a flat dictionary of numpy arrays."""
        return {
            "pose": self.landmarks_to_numpy(results["pose"], 33),
            "face": self.landmarks_to_numpy(results["face"], 468),
            "left_hand": self.landmarks_to_numpy(results["left_hand"], 21),
            "right_hand": self.landmarks_to_numpy(results["right_hand"], 21)
        }

    def draw(self, frame, results):
        """Draws tracked landmarks onto the frame for debugging."""
        annotated_frame = frame.copy()
        # Face
        self.mp_draw.draw_landmarks(
            annotated_frame, results["face"], self.mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
        # Pose
        self.mp_draw.draw_landmarks(
            annotated_frame, results["pose"], self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        # Hands
        self.mp_draw.draw_landmarks(annotated_frame, results["left_hand"], self.mp_holistic.HAND_CONNECTIONS)
        self.mp_draw.draw_landmarks(annotated_frame, results["right_hand"], self.mp_holistic.HAND_CONNECTIONS)
        return annotated_frame
