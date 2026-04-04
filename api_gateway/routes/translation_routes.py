"""
Translation Routes — API endpoints for text translation.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()


class TextInput(BaseModel):
    text: str


class GlossInput(BaseModel):
    gloss: List[str]


@router.post("/text-to-sign")
async def text_to_sign(data: TextInput):
    """Convert text to sign language gloss."""
    from services.translation_service import TranslationServiceHandler
    return TranslationServiceHandler.text_to_sign(data.text)


@router.post("/sign-to-text")
async def sign_to_text(data: GlossInput):
    """Convert sign gloss tokens to text."""
    from services.translation_service import TranslationServiceHandler
    return TranslationServiceHandler.sign_to_text(data.gloss)


@router.post("/batch")
async def batch_translate(texts: List[str]):
    """Batch translate multiple sentences."""
    from services.translation_service import TranslationServiceHandler
    return TranslationServiceHandler.batch(texts)
