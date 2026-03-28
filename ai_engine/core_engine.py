"""
Core Engine

High-level orchestrator that wires together:
- vision feature extraction
- gesture recognition
- text/gloss translation
- motion generation and avatar packaging
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_engine.utils.logger import get_logger


logger = get_logger("ai_engine.core")


class SignVerseEngine:
    """
    A single entrypoint for in-process usage.

    Note: services (`api_gateway`, `api_server`, `avatar_animation`) each expose
    their own APIs; this engine is a convenient composition layer.
    """

    def __init__(self, model_dir: str = "models", fps: int = 30):
        self.model_dir = model_dir
        self.fps = int(fps)

        self._loader = None
        self._realtime = None
        self._translator = None
        self._avatar = None

    @property
    def loader(self):
        if self._loader is None:
            from api_server.model_loader import ModelLoader
            self._loader = ModelLoader(model_dir=self.model_dir)
        return self._loader

    @property
    def realtime(self):
        if self._realtime is None:
            from api_server.realtime_inference import RealtimeInference
            self._realtime = RealtimeInference(self.loader)
        return self._realtime

    @property
    def translator(self):
        if self._translator is None:
            from nlp_translation.inference import TranslationInference
            self._translator = TranslationInference()
        return self._translator

    @property
    def avatar(self):
        if self._avatar is None:
            from avatar_animation.animation_engine import AnimationEngine
            self._avatar = AnimationEngine(fps=self.fps)
        return self._avatar

    def speech_to_sign(self, audio_path: str) -> Dict[str, Any]:
        from ai_engine.inference_pipeline import InferencePipeline
        pipeline = InferencePipeline()
        return pipeline.speech_to_sign(audio_path)

    def sign_to_text_to_speech(self, sequence: List[List[float]]) -> Dict[str, Any]:
        text = self.realtime.sign_to_text(sequence)
        from ai_engine.modules.text_to_speech import TextToSpeech
        tts = TextToSpeech()
        out = tts.synthesize(text)
        return {"text": text, "speech_b64": out.get("audio_b64"), "audio_format": "wav"}

    def speech_to_text_to_sign(self, audio_path: str) -> Dict[str, Any]:
        from ai_engine.modules.speech_to_text import SpeechToText
        from ai_engine.modules.text_to_sign import TextToSignConverter
        stt = SpeechToText()
        t2s = TextToSignConverter()
        text = stt.transcribe(audio_path)
        tokens = t2s.convert(text)
        return {"text": text, "sign_tokens": tokens}

    def text_to_gloss(self, text: str) -> Dict[str, Any]:
        return self.translator.text_to_sign(text)

    def gloss_to_text(self, gloss_tokens: List[str]) -> Dict[str, Any]:
        text = self.translator.sign_to_text(gloss_tokens)
        return {"text": text, "tokens": gloss_tokens}

    def frame_to_features(self, image_bytes: bytes) -> Dict[str, Any]:
        features = self.realtime.extract_features_from_bytes(image_bytes)
        return {"features": features.tolist(), "dim": int(len(features))}

    def keypoints_to_gesture(self, keypoints: List[float]) -> Dict[str, Any]:
        return self.realtime.classify_gesture(keypoints)

    def tokens_to_avatar(self, tokens: List[str], motion: Optional[Any] = None) -> Dict[str, Any]:
        return self.avatar.build_unity_payload(tokens=tokens, motion_sequence=motion)
