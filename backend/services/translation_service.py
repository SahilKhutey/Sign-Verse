from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
import sys
import os
import json
import base64
import httpx
import asyncio

# Add project root to sys.path to allow importing core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from database.db_connection import get_db
from database.models import TranslationHistory
from utils.authentication import get_current_user
from utils.cache_manager import cache_manager
from config import INFERENCE_API_URL

router = APIRouter()

_HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_RETRIES = 2

async def _post_json(path: str, payload: dict):
    url = INFERENCE_API_URL.rstrip("/") + path
    last_error = None
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for attempt in range(_RETRIES + 1):
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
            except httpx.RequestError as e:
                last_error = e
                if attempt < _RETRIES:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
    raise HTTPException(status_code=502, detail=f"Inference API error: {last_error}")

async def _post_audio(path: str, audio_bytes: bytes, filename: str = "audio.wav"):
    url = INFERENCE_API_URL.rstrip("/") + path
    files = {"file": (filename, audio_bytes, "application/octet-stream")}
    last_error = None
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for attempt in range(_RETRIES + 1):
            try:
                resp = await client.post(url, files=files)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
            except httpx.RequestError as e:
                last_error = e
                if attempt < _RETRIES:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
    raise HTTPException(status_code=502, detail=f"Inference API error: {last_error}")

async def verify_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    try:
        return get_current_user(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

class TranslationCreate(BaseModel):
    user_id: int
    input_text: str
    sign_gloss: str
    translation_type: str


class TranslationResponse(BaseModel):
    id: int
    input_text: str
    sign_gloss: str
    translation_type: str

    class Config:
        from_attributes = True


@router.post("/")
async def save_translation(data: TranslationCreate, db: Session = Depends(get_db)):
    record = TranslationHistory(
        user_id=data.user_id,
        input_text=data.input_text,
        sign_gloss=data.sign_gloss,
        translation_type=data.translation_type
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "status": "saved"}


@router.get("/history/{user_id}", response_model=List[TranslationResponse])
async def get_history(user_id: int, db: Session = Depends(get_db), current_user: dict = Depends(verify_auth)):
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    
    # Try Cache
    cache_key = f"history_user_{user_id}"
    cached_data = cache_manager.get(cache_key)
    if cached_data:
        # Assuming cached_data is a JSON string of list of dicts
        return [TranslationResponse(**item) for item in json.loads(cached_data)]
    
    history = db.query(TranslationHistory).filter(TranslationHistory.user_id == user_id).order_by(TranslationHistory.created_at.desc()).limit(50).all()
    
    # Convert SQLAlchemy objects to dictionaries for caching
    history_dicts = [
        {
            "id": h.id,
            "input_text": h.input_text,
            "sign_gloss": h.sign_gloss,
            "translation_type": h.translation_type
        }
        for h in history
    ]

    # Save to Cache
    cache_manager.set(cache_key, history_dicts)
    
    return [TranslationResponse(**item) for item in history_dicts]


class TranslationRequest(BaseModel):
    user_id: int
    data: str  # Base64 encoded video/audio or raw text
    input_type: str = "sign"  # "sign", "text", "audio"
    target_lang: str = "en"

@router.post("/translate")
async def translate(request: TranslationRequest, db: Session = Depends(get_db), current_user: dict = Depends(verify_auth)):
    """Live translation via the AI Engine (Protected)."""
    try:
        # Enforce user_id from token
        if request.user_id != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized for this user ID")

        if request.input_type == "sign":
            # Expect JSON list of keypoint frames for now
            try:
                sequence = json.loads(request.data)
                if not isinstance(sequence, list):
                    raise ValueError("sequence must be a JSON list")
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="For input_type=sign, data must be a JSON list of keypoint frames."
                )
            result = await _post_json("/translate/sign-to-text", {"sequence": sequence})
        elif request.input_type == "audio":
            try:
                audio_bytes = base64.b64decode(request.data, validate=True)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="For input_type=audio, data must be base64-encoded bytes."
                )
            result = await _post_audio("/speech-to-sign/", audio_bytes)
        else:
            result = await _post_json("/translate/text-to-sign", {"text": request.data})
        
        # Optionally save to history
        history = TranslationHistory(
            user_id=request.user_id,
            input_text=request.data[:255], # Truncate for DB
            sign_gloss=result.get("text", "") or " ".join(result.get("tokens", []) or []),
            translation_type=request.input_type
        )
        db.add(history)
        db.commit()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
