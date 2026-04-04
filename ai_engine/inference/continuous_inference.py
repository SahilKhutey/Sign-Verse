"""
Continuous Sign Language Inference Pipeline

Production pipeline:
    Camera → Frame Buffer (30 frames) → CNN Feature Extraction
    → LSTM/Transformer → CTC Decoder → Sentence Output

Flow:
    Mobile Camera
         ↓
    Gesture Detection (MediaPipe)
         ↓
    Pose Sequence (30 frames)
         ↓
    Deep Sign Recognition Model
         ↓
    Sentence Decoder
         ↓
    Text Output
         ↓
    Speech Engine
"""

import cv2
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.ctc_decoder import decode_predictions


class ContinuousInference:
    """
    Real-time continuous sign language inference.
    Buffers 30 frames, runs model inference, and decodes to sentences.
    """

    def __init__(self, model_path, labels, buffer_size=30):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

        self.labels = labels
        self.buffer_size = buffer_size
        self.buffer = []

    def preprocess(self, frame):
        """Resize and normalize frame for model input."""

        frame = cv2.resize(frame, (224, 224))

        frame = frame.astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std

        # HWC → CHW
        frame = np.transpose(frame, (2, 0, 1))

        return torch.tensor(frame).float()

    def process_frame(self, frame):
        """
        Add a frame to the buffer. When buffer is full,
        run inference and return decoded sentence.

        Returns:
            Decoded sentence string, or None if buffer not yet full.
        """

        processed = self.preprocess(frame)
        self.buffer.append(processed)

        if len(self.buffer) >= self.buffer_size:

            video_tensor = torch.stack(self.buffer).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(video_tensor)

            sentences = decode_predictions(logits, self.labels)

            self.buffer.clear()

            return sentences[0] if sentences else None

        return None


def run_live(model_path, labels):
    """Run continuous inference from webcam."""

    engine = ContinuousInference(model_path, labels)

    cap = cv2.VideoCapture(0)

    current_sentence = ""

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = engine.process_frame(frame)

        if result:
            current_sentence = result
            print(f"Recognized: {result}")

        if current_sentence:
            cv2.putText(
                frame,
                current_sentence,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

        cv2.imshow("Continuous Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
