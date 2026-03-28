"""
Request Models — Pydantic schemas for API request validation.
"""

from pydantic import BaseModel
from typing import List, Optional


class SpeechRequest(BaseModel):
    audio_url: Optional[str] = None


class GestureRequest(BaseModel):
    keypoints: List[float]
    timestamp: Optional[float] = None


class TranslationRequest(BaseModel):
    text: str
    source_language: str = "en"
    target_format: str = "gloss"


class BatchTranslationRequest(BaseModel):
    sentences: List[str]


class AnimationRequest(BaseModel):
    tokens: List[str]
    speed: float = 1.0
