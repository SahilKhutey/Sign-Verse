"""
Real-Time Inference — Orchestrates all AI layers per request.
"""

import numpy as np
import cv2

from common.keypoint_schema import FEATURE_DIM_225, TWO_HANDS_DIM, HANDS_SLICE_225


class RealtimeInference:

    def __init__(self, loader):
        self.loader = loader
        self._feature_extractor = None
        self._translator = None

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            from vision_pipeline.feature_extractor import FeatureExtractor
            self._feature_extractor = FeatureExtractor()
        return self._feature_extractor

    @property
    def translator(self):
        if self._translator is None:
            from nlp_translation.inference import TranslationInference
            self._translator = TranslationInference()
        return self._translator

    def extract_features_from_bytes(self, image_bytes):
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return np.zeros(225)
        features, _, _ = self.feature_extractor.extract(frame)
        return features

    def classify_gesture(self, keypoints):
        import torch
        model = self.loader.get_gesture_model()
        if not keypoints:
            return {"gesture": None}
        kp = np.array(keypoints, dtype=np.float32).reshape(-1)

        # Prefer the model's configured input size (fallback to 126).
        expected_dim = getattr(getattr(model, "lstm", None), "input_size", TWO_HANDS_DIM)
        expected_dim = int(expected_dim or TWO_HANDS_DIM)

        # If we receive canonical 225-dim vectors but the model expects hands-only
        # (126 dims), slice out the hands portion.
        if kp.shape[0] == FEATURE_DIM_225 and expected_dim == TWO_HANDS_DIM:
            kp = kp[HANDS_SLICE_225]

        if kp.shape[0] < expected_dim:
            kp = np.pad(kp, (0, expected_dim - kp.shape[0]))
        kp = kp[:expected_dim]
        # Expand to (1, seq, expected_dim) using single frame
        seq = torch.tensor(kp).unsqueeze(0).unsqueeze(0).expand(1, 30, expected_dim)
        with torch.no_grad():
            logits = model(seq)
        pred = logits.argmax(1).item()
        return {"gesture_id": pred}

    def sign_to_text(self, sequence):
        if not sequence:
            return ""
        import torch
        model = self.loader.get_sign_transformer()
        seq = torch.tensor(sequence, dtype=torch.float32)
        if seq.ndim == 2:
            seq = seq.unsqueeze(0)
        # Pad/truncate to model feature dim.
        feature_dim = getattr(model, "feature_dim", seq.size(-1))

        # Canonical 225-dim layout is [body, left_hand, right_hand]. If the model
        # is hands-only (126 dims), slice out hands rather than taking the first 126.
        if seq.size(-1) == FEATURE_DIM_225 and int(feature_dim) == TWO_HANDS_DIM:
            seq = seq[..., HANDS_SLICE_225]

        if seq.size(-1) < feature_dim:
            pad = feature_dim - seq.size(-1)
            seq = torch.nn.functional.pad(seq, (0, pad))
        elif seq.size(-1) > feature_dim:
            seq = seq[..., :feature_dim]

        token_batches = model.translate(seq)
        token_ids = token_batches[0] if token_batches else []

        # Decode token IDs to readable text if a vocab exists.
        try:
            tokenizer = self.loader.get_sign_tokenizer()
            gloss = tokenizer.decode(token_ids)
            if gloss:
                try:
                    return self.translator.sign_to_text(gloss.split())
                except Exception:
                    from nlp_translation.gloss_to_text import gloss_to_text
                    return gloss_to_text(gloss.split())
            return gloss
        except Exception:
            return " ".join(f"TOKEN_{t}" for t in token_ids)

    def text_to_sign(self, text):
        out = self.translator.text_to_sign(text)
        return out.get("gloss") if isinstance(out, dict) else out

    def generate_motion(self, tokens, frames=30):
        pipeline = self.loader.get_diffusion_pipeline()
        motion = pipeline.generate(seq_len=frames)
        return motion[0]

    def process_stream_frame(self, data):
        """Process a single WebSocket frame payload."""
        keypoints = data.get("keypoints", [])
        result = self.classify_gesture(keypoints)
        gesture_id = result.get("gesture_id")
        gesture_label = None
        try:
            gesture_label = self.loader.gesture_label(gesture_id)
        except Exception:
            gesture_label = None

        tokens = self.text_to_sign(data.get("text", ""))
        if (not tokens) and gesture_label:
            tokens = [gesture_label]
        return {
            "gesture_id": gesture_id,
            "gesture_label": gesture_label,
            "sign_tokens": tokens,
            "frame_id": data.get("frame_id", 0)
        }
