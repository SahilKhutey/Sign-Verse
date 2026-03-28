"""
Speech Routes — API endpoints for speech-to-sign translation.
"""

from fastapi import APIRouter, UploadFile
import shutil
import os

router = APIRouter()


@router.post("/transcribe")
async def transcribe_speech(file: UploadFile):
    """Upload audio file and get text transcription."""
    os.makedirs("temp", exist_ok=True)
    file_location = f"temp/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Import here to avoid circular deps
    from services.speech_service import SpeechServiceHandler
    result = SpeechServiceHandler.transcribe(file_location)

    return result


@router.post("/speech-to-sign")
async def speech_to_sign(file: UploadFile):
    """Upload audio and get sign language tokens."""
    os.makedirs("temp", exist_ok=True)
    file_location = f"temp/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from services.speech_service import SpeechServiceHandler
    result = SpeechServiceHandler.speech_to_sign(file_location)

    return result
