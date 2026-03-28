import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """
    High-performance hand tracking using MediaPipe.
    Optimized for real-time inference (~10ms per frame).

    Output: 63 values (21 landmarks × x,y,z)
    """

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def extract_keypoints(self, frame):
        """
        Extract 21 hand landmarks as a flat numpy array of 63 values.
        Returns None if no hand is detected.
        """

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        keypoints = []

        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])

        return np.array(keypoints)
