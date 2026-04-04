import asyncio
import base64
import os
import tempfile
import time
import numpy as np
from caching.translation_cache import TranslationCache
from core.emotion_analyzer import EmotionAnalyzer

class SignVerseTranslationEngine:
    """
    Central orchestration engine for SignVerse AI translation services.
    Bridges Sign-to-Text, Text-to-Sign, STT, and TTS.
    """

    def __init__(self):
        # In a real microservices env, these would be clients to other services
        self.sessions = {}
        self.cache = TranslationCache()
        self.emotion_analyzer = EmotionAnalyzer()
        self._speech_to_text = None
        self._text_to_sign = None
        self._text_to_speech = None
        self._realtime = None

    @property
    def speech_to_text(self):
        if self._speech_to_text is None:
            from ai_engine.modules.speech_to_text import SpeechToText
            self._speech_to_text = SpeechToText()
        return self._speech_to_text

    @property
    def text_to_sign(self):
        if self._text_to_sign is None:
            from ai_engine.modules.text_to_sign import TextToSignConverter
            self._text_to_sign = TextToSignConverter()
        return self._text_to_sign

    @property
    def text_to_speech(self):
        if self._text_to_speech is None:
            from ai_engine.modules.text_to_speech import TextToSpeech
            self._text_to_speech = TextToSpeech()
        return self._text_to_speech

    @property
    def realtime(self):
        if self._realtime is None:
            from api_server.model_loader import ModelLoader
            from api_server.realtime_inference import RealtimeInference
            loader = ModelLoader()
            self._realtime = RealtimeInference(loader)
        return self._realtime

    async def start_conversation_session(self, user_id: str):
        """Initialize a new translation session for a user."""
        self.sessions[user_id] = {
            "start_time": time.time(),
            "frames_processed": 0,
            "last_translation": ""
        }
        return self.sessions[user_id]

    async def process_frame(self, session, frame_data: bytes):
        """
        Process a single video frame for Sign -> Speech translation.
        1. Extract features (Sign)
        2. Translate to Text
        3. Convert Text to Speech
        """
        await asyncio.sleep(0.01)

        session["frames_processed"] = session.get("frames_processed", 0) + 1

        # Extract features and convert to text using realtime pipeline
        features = self.realtime.extract_features_from_bytes(frame_data)
        text = self.realtime.sign_to_text([features.tolist()])

        # TTS
        tts = self.text_to_speech.synthesize(text)

        emotion_result = self.emotion_analyzer.analyze(np.zeros((480, 640, 3)))

        return {
            "text": text,
            "speech_b64": tts.get("audio_b64"),
            "confidence": 0.9,
            "sentiment": emotion_result["dominant"],
            "emotions": emotion_result["emotions"],
        }

    async def translate_sign_to_speech(self, video_data):
        """High-level method for sign to speech translation."""
        # Check cache
        cached = self.cache.get(video_data, input_type="sign")
        if cached: return cached

        result = await self.process_frame({"frames_processed": 0}, video_data)
        self.cache.set(video_data, result, input_type="sign")
        return result

    async def translate_speech_to_sign(self, audio_data):
        """High-level method for speech to sign translation."""
        # Check cache
        cached = self.cache.get(audio_data, input_type="audio")
        if cached: return cached

        result = await self.speech_to_sign(audio_data)
        self.cache.set(audio_data, result, input_type="audio")
        return result

    async def speech_to_sign(self, audio_data: bytes):
        """
        Convert Speech -> Sign Gesture Tokens.
        1. STT (Speech to Text)
        2. Text to Gesture tokens
        """
        await asyncio.sleep(0.01)
        audio_path = None
        if isinstance(audio_data, (bytes, bytearray)):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = tmp.name
            tmp.write(audio_data)
            tmp.close()
        else:
            audio_path = str(audio_data)

        text = self.speech_to_text.transcribe(audio_path)
        tokens = self.text_to_sign.convert(text)

        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        return {
            "text": text,
            "tokens": tokens
        }

    async def sign_to_text_to_speech(self, sequence):
        """
        Full round-trip: Sign -> Text -> Speech (base64 WAV).
        """
        if not sequence:
            return {"text": "", "speech_b64": None}
        text = self.realtime.sign_to_text(sequence)
        tts = self.text_to_speech.synthesize(text)
        return {"text": text, "speech_b64": tts.get("audio_b64")}

    async def speech_to_text_to_sign(self, audio_bytes):
        """
        Full round-trip: Speech -> Text -> Sign tokens.
        """
        return await self.speech_to_sign(audio_bytes)
