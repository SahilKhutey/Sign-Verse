"""
API Server — Real-time Inference Server

Unified FastAPI server exposing all 5 AI layers:
    1. Vision Layer         — /vision/extract
    2. Gesture Layer        — /gesture/classify
    3. Language Layer       — /translate/sign-to-text
    4. Generation Layer     — /generate/motion
    5. AR/VR Interface      — WebSocket /ws/stream
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
from typing import List
import uvicorn
import os
import shutil
import time
from dotenv import load_dotenv

from api_server.model_loader import ModelLoader
from api_server.realtime_inference import RealtimeInference

# Load environment (optional .env)
load_dotenv()

app = FastAPI(
    title="SignVerse AI Server",
    version="2.0.0",
    description="Multimodal Sign Language AI — 5-layer inference API"
)

def _parse_origins(value: str):
    if not value:
        return []
    return [o.strip() for o in value.split(",") if o.strip()]

ALLOWED_ORIGINS = _parse_origins(os.getenv("ALLOWED_ORIGINS", "*"))
allow_all = "*" in ALLOWED_ORIGINS if ALLOWED_ORIGINS else False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else (ALLOWED_ORIGINS or []),
    allow_credentials=False if allow_all else True,
    allow_methods=["*"],
    allow_headers=["*"]
)

loader = ModelLoader()
realtime = RealtimeInference(loader)
ws_clients = []

# Speech-to-sign pipeline (same behavior as api_gateway)
_speech_pipeline = None
os.makedirs("temp", exist_ok=True)


def _get_speech_pipeline():
    global _speech_pipeline
    if _speech_pipeline is None:
        from ai_engine.inference_pipeline import InferencePipeline
        _speech_pipeline = InferencePipeline()
    return _speech_pipeline


@app.get("/health")
async def health():
    return {"status": "ok", "models": loader.loaded_models()}


@app.get("/health/ready")
async def health_ready():
    try:
        loader.get_gesture_model()
        loader.get_sign_transformer()
        loader.get_diffusion_pipeline()
        return {"status": "ready", "models": loader.loaded_models()}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or os.urandom(8).hex()
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    response.headers["x-request-id"] = request_id
    try:
        from common.logger import get_logger
        logger = get_logger("signverse-api-server")
        logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else None,
            },
        )
    except Exception:
        pass
    return response


@app.post("/vision/extract")
async def extract_features(file: UploadFile):
    """Extract pose features from uploaded image/video frame."""
    contents = await file.read()
    features = realtime.extract_features_from_bytes(contents)
    return {"features": features.tolist(), "dim": len(features)}


@app.post("/gesture/classify")
async def classify_gesture(data: KeypointRequest):
    """Classify gesture from keypoint vector."""
    keypoints = data.keypoints
    result = realtime.classify_gesture(keypoints)
    gesture_id = result.get("gesture_id")
    label = loader.gesture_label(gesture_id)
    if label:
        result["gesture_label"] = label
    return result


@app.post("/speech-to-sign/")
async def speech_to_sign(file: UploadFile):
    """
    Accepts an audio file upload and returns:
    {
        "text": "hello how are you",
        "sign_tokens": ["HELLO", "HOW", "YOU"]
    }
    """
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio file")

    file_location = os.path.join("temp", file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline = _get_speech_pipeline()
    return pipeline.speech_to_sign(file_location)


@app.post("/translate/speech-to-sign")
async def translate_speech_to_sign(file: UploadFile):
    """Alias for speech-to-sign translation."""
    return await speech_to_sign(file)


@app.post("/analyze/frame")
async def analyze_frame(file: UploadFile, return_keypoints: bool = False):
    """
    Single-shot helper: image frame -> keypoints -> gesture classification.

    Returns:
      - gesture_id
      - gesture_label (if label_map.json is available)
      - dim
      - sign_tokens (best-effort: [gesture_label] if known)
      - keypoints (optional, when return_keypoints=true)
    """
    contents = await file.read()
    features = realtime.extract_features_from_bytes(contents)
    result = realtime.classify_gesture(features.tolist())

    gesture_id = result.get("gesture_id")
    label = loader.gesture_label(gesture_id)
    tokens = [label] if label else []

    out = {
        "gesture_id": gesture_id,
        "gesture_label": label,
        "sign_tokens": tokens,
        "dim": int(len(features)),
    }
    if return_keypoints:
        out["keypoints"] = features.tolist()
    return out


@app.post("/translate/sign-to-text")
async def sign_to_text(data: SignToTextRequest):
    """Translate sign keypoint sequence to text."""
    sequence = data.sequence
    text = realtime.sign_to_text(sequence)
    return {"text": text}


@app.post("/translate/sign-to-speech")
async def sign_to_speech(data: SignToTextRequest):
    """
    Sign -> Text -> Speech (base64 WAV).
    """
    sequence = data.sequence
    text = realtime.sign_to_text(sequence)
    try:
        from ai_engine.modules.text_to_speech import TextToSpeech
        tts = TextToSpeech()
        out = tts.synthesize(text)
        return {"text": text, "speech_b64": out.get("audio_b64"), "audio_format": "wav"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@app.post("/translate/text-to-sign")
async def text_to_sign(data: TextToSignRequest):
    """Convert text to sign gloss tokens."""
    text = data.text
    tokens = realtime.text_to_sign(text)
    return {"tokens": tokens}


@app.post("/generate/motion")
async def generate_motion(data: MotionRequest):
    """Generate 3D motion from sign tokens."""
    tokens = data.tokens
    frames = data.frames
    motion = realtime.generate_motion(tokens, frames)
    return {"motion": motion.tolist(), "frames": len(motion)}


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """WebSocket for real-time gesture streaming to AR/VR clients."""
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_json()
            result = realtime.process_stream_frame(data)
            await ws.send_json(result)
    except WebSocketDisconnect:
        ws_clients.remove(ws)


if __name__ == "__main__":
    reload = os.getenv("UVICORN_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("api_server.server:app", host="0.0.0.0", port=8000, reload=reload)
class KeypointRequest(BaseModel):
    keypoints: conlist(float, min_length=1) = Field(..., description="Flattened keypoint vector")


class SignToTextRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Sequence of keypoint frames")


class TextToSignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class MotionRequest(BaseModel):
    tokens: List[str] = Field(default_factory=list)
    frames: int = Field(default=30, ge=1, le=240)
