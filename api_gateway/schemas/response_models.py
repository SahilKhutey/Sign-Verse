"""
Response Models — Pydantic schemas for API responses.
"""

from pydantic import BaseModel
from typing import List, Optional


class TranscriptionResponse(BaseModel):
    text: str


class GestureResponse(BaseModel):
    gesture: Optional[str]
    confidence: float = 0.0


class TranslationResponse(BaseModel):
    text: str
    sign_tokens: List[str]


class AnimationResponse(BaseModel):
    clips: List[str]
    queued: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
