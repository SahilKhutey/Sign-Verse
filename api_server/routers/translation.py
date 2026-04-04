from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/translate", tags=["translation"])

class TranslationRequest(BaseModel):
    text: str

@router.post("/sign-to-text")
async def sign_to_text():
    return {"text": "hello"}

@router.post("/text-to-sign")
async def text_to_sign(data: TranslationRequest):
    return {"tokens": ["HELLO"]}
