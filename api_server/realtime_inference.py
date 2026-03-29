"""
Real-Time Inference — Orchestrates all AI layers per request.
"""

import numpy as np
import cv2
import torch

from vision_system.motion_intelligence import MotionIntelligence

class RealtimeInference:

    def __init__(self, loader):
        self.loader = loader
        self.mi = MotionIntelligence()
        self._translator = None

    @property
    def translator(self):
        if self._translator is None:
            from nlp_translation.inference import TranslationInference
            self._translator = TranslationInference()
        return self._translator

    def extract_features_from_bytes(self, image_bytes):
        """
        Extract high-fidelity motion intelligence features.
        Returns: 848-dim vector (424 geo + 424 vel)
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return np.zeros(848)
        
        # Use the new MotionIntelligence pipeline
        out = self.mi.process_frame(frame)
        return out["motion_vector"]

    def predict_intent(self, intelligence_vector):
        """
        Map motion intelligence to specific UI intents.
        """
        if intelligence_vector is None or len(intelligence_vector) == 0:
            return "IDLE"
        
        # Heuristic: if velocity (last 424 dims) is high, user is SIGNING
        if np.mean(np.abs(intelligence_vector[424:])) > 0.05:
            return "SIGNING"
        return "IDLE"

    def classify_gesture_sequence(self, sequence):
        """
        Classify a full keypoint sequence for temporal recognition.
        """
        model = self.loader.get_gesture_model()
        if not sequence:
            return {"gesture_id": None, "confidence": None}

        seq_np = np.array(sequence, dtype=np.float32)
        if seq_np.ndim == 1:
            seq_np = seq_np.reshape(1, -1)
        
        expected_dim = getattr(getattr(model, "lstm", None), "input_size", 848)
        
        if seq_np.shape[-1] != expected_dim:
            if seq_np.shape[-1] < expected_dim:
                pad = expected_dim - seq_np.shape[-1]
                seq_np = np.pad(seq_np, ((0, 0), (0, pad)))
            else:
                seq_np = seq_np[:, :expected_dim]

        seq = torch.tensor(seq_np).unsqueeze(0)
        with torch.no_grad():
            logits = model(seq)
            probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1).item()
        conf = float(probs[0, pred].item()) if probs.numel() > 0 else None
        return {"gesture_id": pred, "confidence": conf}

    def sign_to_text(self, sequence):
        if not sequence:
            return ""
        model = self.loader.get_sign_transformer()
        seq = torch.tensor(sequence, dtype=torch.float32)
        if seq.ndim == 2:
            seq = seq.unsqueeze(0)
            
        feature_dim = getattr(model, "feature_dim", 848)
        if seq.size(-1) != feature_dim:
             if seq.size(-1) < feature_dim:
                 pad = feature_dim - seq.size(-1)
                 seq = torch.nn.functional.pad(seq, (0, pad))
             else:
                 seq = seq[..., :feature_dim]

        token_batches = model.translate(seq)
        token_ids = token_batches[0] if token_batches else []

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

    def generate_motion(self, tokens, frames=30):
        pipeline = self.loader.get_diffusion_pipeline()
        motion = pipeline.generate(seq_len=frames)
        return motion[0]

    def process_stream_frame(self, data):
        """Process a single WebSocket frame payload (now 3D-aware)."""
        mode = data.get("mode", "2d") # "2d" or "3d"
        
        if "image" in data:
            intel_vector = self.extract_features_from_bytes(data["image"])
        else:
            raw = data.get("keypoints", [])
            if isinstance(raw, list) and len(raw) > 0:
                intel_vector = np.array(raw, dtype=np.float32)
                # Reshape/Normalize 3D if needed
                if mode == "3d":
                    # Intel vector is (N_joints, 3) 
                    intel_vector = intel_vector.flatten()
                    
                if len(intel_vector) < 848:
                     pad = 848 - len(intel_vector)
                     intel_vector = np.pad(intel_vector, (0, pad))
                else:
                     intel_vector = intel_vector[:848]
            else:
                intel_vector = np.zeros(848)

        result = self.classify_gesture_sequence([intel_vector])
        intent = self.predict_intent(intel_vector)
        
        gesture_id = result.get("gesture_id")
        gesture_label = None
        try:
            gesture_label = self.loader.gesture_label(gesture_id)
        except Exception:
            gesture_label = "unknown"

        return {
            "gesture_id": gesture_id,
            "gesture_label": gesture_label,
            "intent": intent,
            "intelligence_vector": intel_vector.tolist(),
            "frame_id": data.get("frame_id", 0),
            "mode": mode
        }
