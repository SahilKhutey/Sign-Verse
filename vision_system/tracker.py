import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

class FullBodyTracker:
    def __init__(self):
        self.model = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, results, count):
        if results is None:
            return np.zeros((count, 3))
        return np.array([[lm.x, lm.y, lm.z] for lm in results.landmark])

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.model.process(rgb)

        data = {
            "face": self.extract(res.face_landmarks, 468),
            "left_hand": self.extract(res.left_hand_landmarks, 21),
            "right_hand": self.extract(res.right_hand_landmarks, 21),
            "pose": self.extract(res.pose_landmarks, 33)
        }
        return data
