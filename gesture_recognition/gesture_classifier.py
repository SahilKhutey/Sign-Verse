import numpy as np


class GestureClassifier:

    def __init__(self):

        self.gesture_map = {
            0: "HELLO",
            1: "THANK_YOU",
            2: "YES",
            3: "NO"
        }

    def classify(self, keypoints):
        """
        Classify hand keypoints into a gesture label.
        Placeholder logic — later replace with trained neural network.
        """

        if len(keypoints) == 0:
            return None

        # placeholder inference logic
        features = np.array(keypoints).flatten()

        score = np.sum(features)

        gesture_id = int(score * 10) % 4

        return self.gesture_map[gesture_id]
