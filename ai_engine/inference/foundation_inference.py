"""
Foundation Model Inference Pipeline

Real-time production pipeline:
    Camera → Pose Extraction → Gesture Tokenization
    → Foundation Transformer → Sentence Decoder → Text/Speech

Latency target: <100ms end-to-end

Components:
    1. PoseExtractor     — MediaPipe Holistic (~15ms)
    2. GestureTokenizer  — KMeans lookup (~1ms)
    3. Foundation Model  — Transformer forward pass (~30ms)
    4. Text Decoder      — Autoregressive generation (~50ms)
"""

import cv2
import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class FoundationInference:
    """
    End-to-end inference using the sign language foundation model.
    Buffers pose sequences and translates to text.
    """

    def __init__(self, model_path=None, tokenizer_path=None,
                 text_model_path=None, buffer_size=30):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.pose_buffer = []

        # Lazy-load components
        self._pose_extractor = None
        self._tokenizer = None
        self._model = None
        self._text_model = None

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.text_model_path = text_model_path

    @property
    def pose_extractor(self):
        if self._pose_extractor is None:
            from training.data_pipeline.pose_extractor import PoseExtractor
            self._pose_extractor = PoseExtractor()
        return self._pose_extractor

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from training.data_pipeline.gesture_tokenizer import GestureTokenizer
            self._tokenizer = GestureTokenizer(vocab_path=self.tokenizer_path)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None and self.model_path:
            from models.sign_foundation_transformer import SignFoundationModel
            self._model = SignFoundationModel()
            self._model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def process_frame(self, frame):
        """
        Add frame to pose buffer. Returns translation when buffer is full.
        """
        # Extract pose
        pose_vector = self.pose_extractor.extract_flat_vector(frame)
        self.pose_buffer.append(pose_vector)

        if len(self.pose_buffer) >= self.buffer_size:
            result = self._run_inference()
            self.pose_buffer.clear()
            return result

        return None

    def _run_inference(self):
        """Run full inference pipeline on buffered poses."""
        start_time = time.perf_counter()

        # Tokenize pose sequence
        features = np.array(self.pose_buffer)
        tokens = self.tokenizer.tokenize_sequence(features)

        # Foundation model forward pass
        if self.model is not None:
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(token_tensor)
                predicted_tokens = torch.argmax(logits, dim=-1).squeeze().tolist()
        else:
            predicted_tokens = tokens

        # Decode tokens to labels
        decoded = self.tokenizer.decode(
            predicted_tokens if isinstance(predicted_tokens, list) else [predicted_tokens]
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "gesture_tokens": tokens,
            "predicted": predicted_tokens,
            "decoded": decoded,
            "sentence": " ".join(decoded),
            "latency_ms": round(elapsed_ms, 2)
        }

    def run_live(self):
        """Run real-time inference from webcam."""
        cap = cv2.VideoCapture(0)
        current_text = ""

        print("Foundation Model Inference — Press ESC to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            if result:
                current_text = result["sentence"]
                print(f"[{result['latency_ms']}ms] {current_text}")

            if current_text:
                cv2.putText(
                    frame, current_text,
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2
                )

            # Show buffer progress
            progress = len(self.pose_buffer) / self.buffer_size
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (20, 80), (20 + bar_width, 95), (0, 200, 0), -1)
            cv2.rectangle(frame, (20, 80), (220, 95), (255, 255, 255), 1)

            cv2.imshow("SignVerse Foundation Model", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = FoundationInference(
        model_path=None,  # Set path to trained model
        tokenizer_path=None  # Set path to fitted tokenizer
    )
    engine.run_live()
