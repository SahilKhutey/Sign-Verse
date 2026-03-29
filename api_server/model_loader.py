"""
Model Loader — Lazy-loads all AI models on demand.

Manages model lifecycle:
    - Gesture recognition model
    - Sign transformer
    - Multimodal LLM
    - Gesture diffusion model
"""

from __future__ import annotations

import glob
import csv
import json
import os
import re
from typing import Any, Dict, Optional

import torch

from nlp_translation.tokenizer import SignTokenizer


def _find_latest_checkpoint(model_dir: str, prefix: str) -> Optional[str]:
    """
    Match CheckpointManager.find_latest() behavior:
    1) prefer {prefix}_best.pt
    2) otherwise pick highest-epoch {prefix}_epoch*.pt
    """
    best_path = os.path.join(model_dir, f"{prefix}_best.pt")
    if os.path.exists(best_path):
        return best_path

    pattern = os.path.join(model_dir, f"{prefix}_epoch*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _epoch_num(p: str) -> int:
        name = os.path.basename(p)
        try:
            part = name.replace(f"{prefix}_epoch", "").replace(".pt", "")
            return int(part)
        except Exception:
            return 0

    candidates.sort(key=_epoch_num, reverse=True)
    return candidates[0]


def _extract_model_state(obj: Any) -> Any:
    """Support both raw state_dict checkpoints and BaseTrainer payload checkpoints."""
    if isinstance(obj, dict) and isinstance(obj.get("model_state"), dict):
        return obj["model_state"]
    return obj


def _choose_nhead(d_model: int, preferred: int = 8) -> int:
    """
    Choose a number of heads that divides d_model.
    """
    if not d_model or d_model <= 0:
        return preferred
    
    # Try common head counts
    for h in [preferred, 16, 12, 10, 8, 6, 4, 2, 1]:
        if h <= d_model and d_model % h == 0:
            # Additional check: head_dim should be reasonable (e.g. >= 8)
            if d_model // h >= 8:
                return int(h)
    
    # Fallback to anything that divides
    for h in range(d_model, 0, -1):
        if d_model % h == 0:
            return h
    return 1


def _infer_transformer_num_layers(state: Dict[str, Any], prefix: str) -> Optional[int]:
    # Example keys: transformer.layers.0.self_attn.in_proj_weight
    idxs = set()
    pfx = prefix + "."
    for k in state.keys():
        if not k.startswith(pfx):
            continue
        rest = k[len(pfx):]
        idx = rest.split(".", 1)[0]
        if idx.isdigit():
            idxs.add(int(idx))
    if not idxs:
        return None
    return max(idxs) + 1


def _infer_lstm_num_layers(state: Dict[str, Any]) -> Optional[int]:
    # Keys: lstm.weight_ih_l0, lstm.weight_ih_l0_reverse, lstm.weight_ih_l1, ...
    idxs = set()
    pfx = "lstm.weight_ih_l"
    for k in state.keys():
        if not k.startswith(pfx):
            continue
        rest = k[len(pfx):]
        m = re.match(r"^(\\d+)", rest)
        if m:
            idxs.add(int(m.group(1)))
    if not idxs:
        return None
    return max(idxs) + 1


def _load_state_dict_forgiving(model: torch.nn.Module, state: Dict[str, Any]) -> None:
    """
    Load a checkpoint while skipping shape-mismatched tensors (common when class/vocab
    counts change). This keeps inference usable instead of crashing.
    """
    if not isinstance(state, dict):
        return

    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k not in model_state:
            continue
        try:
            if hasattr(v, "shape") and hasattr(model_state[k], "shape") and model_state[k].shape != v.shape:
                continue
        except Exception:
            pass
        filtered[k] = v

    model.load_state_dict(filtered, strict=False)


class ModelLoader:

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._models: Dict[str, Any] = {}
        self._gesture_id_to_label: Optional[Dict[int, str]] = None
        self._sign_tokenizer: Optional[SignTokenizer] = None
        self._video_sign_tokenizer: Optional[SignTokenizer] = None

    def loaded_models(self):
        return list(self._models.keys())

    def get_gesture_model(self):
        if "gesture" not in self._models:
            from ai_models.gesture_recognition.model import GestureModel

            ckpt_path = _find_latest_checkpoint(self.model_dir, "gesture_model")

            input_size = 126
            hidden_size = 256
            num_classes = 2000
            num_lstm_layers = 2
            num_transformer_layers = 4

            state = None
            if ckpt_path and os.path.exists(ckpt_path):
                payload = torch.load(ckpt_path, map_location="cpu")
                state = _extract_model_state(payload)

                if isinstance(state, dict):
                    try:
                        w = state.get("lstm.weight_ih_l0")
                        if hasattr(w, "shape"):
                            input_size = int(w.shape[1])
                    except Exception:
                        pass
                    try:
                        w = state.get("lstm_proj.weight")
                        if hasattr(w, "shape"):
                            hidden_size = int(w.shape[0])
                    except Exception:
                        pass
                    try:
                        w = state.get("fc.3.weight")
                        if hasattr(w, "shape"):
                            num_classes = int(w.shape[0])
                    except Exception:
                        pass

                    nl = _infer_lstm_num_layers(state)
                    if nl is not None:
                        num_lstm_layers = int(nl)
                    nt = _infer_transformer_num_layers(state, "transformer.layers")
                    if nt is not None:
                        num_transformer_layers = int(nt)
            else:
                # If we have label metadata, use it as a better default.
                id_to_label = self.get_gesture_id_to_label()
                if id_to_label:
                    num_classes = int(max(id_to_label.keys()) + 1)

            nhead = _choose_nhead(hidden_size, preferred=8)
            model = GestureModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_lstm_layers=num_lstm_layers,
                nhead=nhead,
                num_transformer_layers=num_transformer_layers,
            )

            if isinstance(state, dict):
                _load_state_dict_forgiving(model, state)

            model.eval()
            self._models["gesture"] = model

        return self._models["gesture"]

    def get_asl_cnn_model(self):
        """
        Optional Keras CNN model for letter-level ASL image classification.
        """
        if "asl_cnn" not in self._models:
            from ai_models.gesture_recognition.cnn_asl import ASLCNNClassifier

            model_path = os.path.join(self.model_dir, "asl_cnn.h5")
            labels_path = os.path.join(self.model_dir, "asl_cnn_labels.json")
            self._models["asl_cnn"] = ASLCNNClassifier(
                model_path=model_path,
                labels_path=labels_path if os.path.exists(labels_path) else None,
            )
        return self._models["asl_cnn"]

    def get_video_lstm_model(self):
        """
        Optional video CNN+LSTM model for isolated sign recognition.
        """
        if "video_lstm" not in self._models:
            from ai_models.gesture_recognition.video_lstm_keras import VideoLSTMClassifier

            model_path = os.path.join(self.model_dir, "video_lstm.h5")
            labels_path = os.path.join(self.model_dir, "video_lstm_labels.json")
            self._models["video_lstm"] = VideoLSTMClassifier(
                model_path=model_path,
                labels_path=labels_path if os.path.exists(labels_path) else None,
                max_frames=30,
            )
        return self._models["video_lstm"]

    def get_sign_transformer(self):
        if "sign_transformer" not in self._models:
            from ai_models.sign_transformer.train_transformer import SignTransformer

            vocab_path = os.path.join(self.model_dir, "sign_transformer_vocab.json")
            tokenizer = None
            if os.path.exists(vocab_path):
                try:
                    tokenizer = SignTokenizer(vocab_path=vocab_path)
                except Exception:
                    tokenizer = None

            ckpt_path = _find_latest_checkpoint(self.model_dir, "sign_transformer")

            feature_dim = 126
            d_model = 512
            num_layers = 6
            text_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 10000)

            state = None
            if ckpt_path and os.path.exists(ckpt_path):
                payload = torch.load(ckpt_path, map_location="cpu")
                state = _extract_model_state(payload)

                if isinstance(state, dict):
                    # Encoder input projection provides feature_dim + d_model.
                    try:
                        w = state.get("encoder.input_proj.weight")
                        if hasattr(w, "shape"):
                            d_model = int(w.shape[0])
                            feature_dim = int(w.shape[1])
                    except Exception:
                        pass

                    # Decoder embedding provides vocab_size + d_model.
                    try:
                        w = state.get("decoder.embedding.weight")
                        if hasattr(w, "shape"):
                            text_vocab_size = int(w.shape[0])
                            d_model = int(w.shape[1])
                    except Exception:
                        pass

                    nl = _infer_transformer_num_layers(state, "encoder.encoder.layers")
                    if nl is not None:
                        num_layers = int(nl)

            nhead = _choose_nhead(d_model, preferred=8)
            model = SignTransformer(
                feature_dim=feature_dim,
                text_vocab_size=text_vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
            )

            if isinstance(state, dict):
                _load_state_dict_forgiving(model, state)

            model.eval()
            self._models["sign_transformer"] = model

            if tokenizer is not None:
                self._sign_tokenizer = tokenizer

        return self._models["sign_transformer"]

    def get_video_sign_transformer(self):
        """
        Return a sign->text transformer intended for video-derived frame features.

        Source preference:
        1) models/video_sign_transformer_{best|epoch*}.pt
        2) fallback to models/sign_transformer_{best|epoch*}.pt
        """
        if "video_sign_transformer" not in self._models:
            from ai_models.sign_transformer.train_transformer import SignTransformer

            vocab_candidates = [
                os.path.join(self.model_dir, "video_sign_transformer_vocab.json"),
                os.path.join(self.model_dir, "sign_transformer_vocab.json"),
            ]
            tokenizer = None
            for vocab_path in vocab_candidates:
                if os.path.exists(vocab_path):
                    try:
                        tokenizer = SignTokenizer(vocab_path=vocab_path)
                        break
                    except Exception:
                        tokenizer = None

            ckpt_path = _find_latest_checkpoint(self.model_dir, "video_sign_transformer")
            if not ckpt_path:
                ckpt_path = _find_latest_checkpoint(self.model_dir, "sign_transformer")

            feature_dim = 225
            d_model = 512
            num_layers = 6
            text_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 10000)

            state = None
            if ckpt_path and os.path.exists(ckpt_path):
                payload = torch.load(ckpt_path, map_location="cpu")
                state = _extract_model_state(payload)

                if isinstance(state, dict):
                    try:
                        w = state.get("encoder.input_proj.weight")
                        if hasattr(w, "shape"):
                            d_model = int(w.shape[0])
                            feature_dim = int(w.shape[1])
                    except Exception:
                        pass

                    try:
                        w = state.get("decoder.embedding.weight")
                        if hasattr(w, "shape"):
                            text_vocab_size = int(w.shape[0])
                            d_model = int(w.shape[1])
                    except Exception:
                        pass

                    nl = _infer_transformer_num_layers(state, "encoder.encoder.layers")
                    if nl is not None:
                        num_layers = int(nl)

            nhead = _choose_nhead(d_model, preferred=8)
            model = SignTransformer(
                feature_dim=feature_dim,
                text_vocab_size=text_vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
            )

            if isinstance(state, dict):
                _load_state_dict_forgiving(model, state)

            model.eval()
            self._models["video_sign_transformer"] = model
            if tokenizer is not None:
                self._video_sign_tokenizer = tokenizer

        return self._models["video_sign_transformer"]

    def get_sign_tokenizer(self) -> SignTokenizer:
        """
        Return the tokenizer used by the sign->text transformer (if trained).

        Source preference:
        1) models/sign_transformer_vocab.json
        """
        if self._sign_tokenizer is not None:
            return self._sign_tokenizer

        vocab_path = os.path.join(self.model_dir, "sign_transformer_vocab.json")
        if os.path.exists(vocab_path):
            try:
                self._sign_tokenizer = SignTokenizer(vocab_path=vocab_path)
                return self._sign_tokenizer
            except Exception:
                pass

        # Fallback: a minimal tokenizer with only special tokens.
        self._sign_tokenizer = SignTokenizer()
        return self._sign_tokenizer

    def get_video_sign_tokenizer(self) -> SignTokenizer:
        """
        Return tokenizer used by video sign->text transformer.

        Source preference:
        1) models/video_sign_transformer_vocab.json
        2) models/sign_transformer_vocab.json
        """
        if self._video_sign_tokenizer is not None:
            return self._video_sign_tokenizer

        candidates = [
            os.path.join(self.model_dir, "video_sign_transformer_vocab.json"),
            os.path.join(self.model_dir, "sign_transformer_vocab.json"),
        ]
        for vocab_path in candidates:
            if os.path.exists(vocab_path):
                try:
                    self._video_sign_tokenizer = SignTokenizer(vocab_path=vocab_path)
                    return self._video_sign_tokenizer
                except Exception:
                    continue

        self._video_sign_tokenizer = SignTokenizer()
        return self._video_sign_tokenizer

    def get_diffusion_pipeline(self):
        if "diffusion" not in self._models:
            from ai_models.gesture_diffusion.diffusion_model import GestureDiffusionPipeline

            ckpt_path = _find_latest_checkpoint(self.model_dir, "diffusion_model")
            pipeline = GestureDiffusionPipeline(
                model_path=ckpt_path if ckpt_path and os.path.exists(ckpt_path) else None
            )
            self._models["diffusion"] = pipeline
        return self._models["diffusion"]

    def get_co_speech_model(self):
        if "co_speech" not in self._models:
            from ai_models.co_speech_generation.audio_to_gesture import AudioToGestureTransformer
            
            ckpt_path = _find_latest_checkpoint(self.model_dir, "co_speech_model")
            
            audio_dim = 128 # Default for audio features
            emotion_dim = 32 # Matches model __init__
            motion_dim = 225
            
            state = None
            if ckpt_path and os.path.exists(ckpt_path):
                payload = torch.load(ckpt_path, map_location="cpu")
                state = _extract_model_state(payload)
                
                if isinstance(state, dict):
                    try:
                        w = state.get("audio_proj.weight") # Check audio_proj instead of audio_encoder
                        if hasattr(w, "shape"):
                            audio_dim = int(w.shape[1])
                    except Exception: pass
                    
            model = AudioToGestureTransformer(
                audio_dim=audio_dim,
                emotion_dim=emotion_dim,
                motion_dim=motion_dim
            )
            
            if isinstance(state, dict):
                _load_state_dict_forgiving(model, state)
            
            model.eval()
            self._models["co_speech"] = model
        return self._models["co_speech"]

    # ---------------------------------------------------------------------
    # Metadata helpers
    # ---------------------------------------------------------------------
    def get_gesture_id_to_label(self):
        """
        Return mapping: gesture_id (int) -> label_name (str).

        Source preference:
        1) models/label_map.json
        2) training-data/label_map.json

        Note: unified preprocessor saves label_map.json as {label_name: label_id},
        so we invert it here.
        """
        if self._gesture_id_to_label is not None:
            return self._gesture_id_to_label

        candidates = [
            os.path.join(self.model_dir, "label_map.json"),
            os.path.join("training-data", "label_map.json"),
        ]

        id_to_label: Dict[int, str] = {}
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    label_to_id = json.load(f) or {}
                # Invert: {label: id} -> {id: label}
                for label, gid in label_to_id.items():
                    try:
                        gid_int = int(gid)
                    except Exception:
                        continue
                    if gid_int not in id_to_label:
                        id_to_label[gid_int] = str(label)
                break
            except Exception:
                continue

        # Fallback: build mapping from labels.csv if no label_map.json exists yet.
        if not id_to_label:
            labels_csv = os.path.join("training-data", "labels.csv")
            if os.path.exists(labels_csv):
                try:
                    with open(labels_csv, "r", newline="") as f:
                        for row in csv.DictReader(f):
                            try:
                                gid_int = int(row.get("label_id", ""))
                            except Exception:
                                continue
                            label = (row.get("label_name") or row.get("label") or row.get("gloss") or "").strip()
                            if label and gid_int not in id_to_label:
                                id_to_label[gid_int] = str(label)
                except Exception:
                    pass

        self._gesture_id_to_label = id_to_label
        return self._gesture_id_to_label

    def gesture_label(self, gesture_id: int):
        if gesture_id is None:
            return None
        try:
            gid = int(gesture_id)
        except Exception:
            return None
        return self.get_gesture_id_to_label().get(gid)
