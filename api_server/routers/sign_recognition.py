from fastapi import APIRouter

router = APIRouter(prefix="/gesture", tags=["recognition"])

@router.post("/classify")
async def classify_gesture():
    return {"gesture_id": 1, "gesture_label": "hello", "confidence": 0.95}
