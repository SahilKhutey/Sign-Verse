"""
Gesture Recognition Inference — Real-Time 30 FPS Pipeline

Pipeline:
    Camera Frame → MediaPipe hand detection → Pose keypoints
    → Feature vector → Gesture model → Predicted sign
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ai_models.gesture_recognition.model import GestureModel


class GestureInference:

    def __init__(self, model_path=None, num_classes=500, seq_len=30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.buffer = []

        state = None
        inferred_num_classes = int(num_classes)
        if model_path and os.path.exists(model_path):
            payload = torch.load(model_path, map_location=self.device)
            state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
            if isinstance(state, dict):
                try:
                    w = state.get("fc.3.weight")
                    if hasattr(w, "shape"):
                        inferred_num_classes = int(w.shape[0])
                except Exception:
                    pass

        self.model = GestureModel(num_classes=inferred_num_classes).to(self.device)
        if isinstance(state, dict):
            # Skip any shape-mismatched tensors (e.g., class count differences).
            cur = self.model.state_dict()
            filtered = {}
            for k, v in state.items():
                if k not in cur:
                    continue
                try:
                    if hasattr(v, "shape") and hasattr(cur[k], "shape") and v.shape != cur[k].shape:
                        continue
                except Exception:
                    pass
                filtered[k] = v
            self.model.load_state_dict(filtered, strict=False)
        self.model.eval()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            model_complexity=0
        )

    def extract_keypoints(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        keypoints = np.zeros(126)

        if result.multi_hand_landmarks:
            for i, hand in enumerate(result.multi_hand_landmarks[:2]):
                offset = i * 63
                for j, lm in enumerate(hand.landmark):
                    keypoints[offset + j*3:offset + j*3 + 3] = [lm.x, lm.y, lm.z]

        return keypoints

    def predict(self, frame):
        kp = self.extract_keypoints(frame)
        self.buffer.append(kp)

        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        if len(self.buffer) == self.seq_len:
            seq = torch.tensor(
                np.array(self.buffer), dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(seq)
                return torch.argmax(logits, dim=1).item()

        return None

    def run_live(self, labels=None):
        cap = cv2.VideoCapture(0)
        print("Gesture Recognition — Press ESC to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pred = self.predict(frame)
            label = labels[pred] if (pred is not None and labels) else f"Class {pred}"

            if pred is not None:
                cv2.putText(frame, label, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
