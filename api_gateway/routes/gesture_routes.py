"""
Gesture Routes — API endpoints for gesture recognition.
"""

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel
from typing import List
import shutil
import os

router = APIRouter()


class KeypointData(BaseModel):
    keypoints: List[float]


@router.post("/classify")
async def classify_gesture(data: KeypointData):
    """Classify gesture from keypoint data."""
    from services.gesture_service import GestureServiceHandler
    result = GestureServiceHandler.classify(data.keypoints)
    return result


@router.post("/detect-from-video")
async def detect_from_video(file: UploadFile):
    """Detect gestures from uploaded video."""
    os.makedirs("temp", exist_ok=True)
    file_location = f"temp/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from services.gesture_service import GestureServiceHandler
    result = GestureServiceHandler.detect_from_video(file_location)

    return result
