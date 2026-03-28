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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR
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

STREAM_TOKEN = os.getenv("STREAM_TOKEN")
STREAM_TOKEN_REQUIRED = os.getenv("STREAM_TOKEN_REQUIRED", "false").lower() in {"1", "true", "yes"}

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


class KeypointRequest(BaseModel):
    keypoints: conlist(float, min_length=1) = Field(..., description="Flattened keypoint vector")


class SignToTextRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Sequence of keypoint frames")


class TextToSignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class MotionRequest(BaseModel):
    tokens: List[str] = Field(default_factory=list)
    frames: int = Field(default=30, ge=1, le=240)


class StreamInitRequest(BaseModel):
    fps: int = Field(default=15, ge=1, le=60)
    window: int = Field(default=8, ge=1, le=60)
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    max_frame_bytes: int = Field(default=1_000_000, ge=10_000, le=5_000_000)
    use_sequence: bool = False
    sequence_window: int = Field(default=30, ge=5, le=120)


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
    request.state.request_id = request_id
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    try:
        from common.logger import get_logger
        logger = get_logger("signverse-api-server")
        logger.exception("unhandled_exception", extra={"request_id": getattr(request.state, "request_id", None)})
    except Exception:
        pass
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "server_error",
            "detail": "internal error",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.post("/vision/extract")
async def extract_features(file: UploadFile):
    """Extract pose features from uploaded image/video frame."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")
    contents = await file.read()
    features = realtime.extract_features_from_bytes(contents)
    return {"features": features.tolist(), "dim": len(features)}


@app.post("/vision/extract-debug")
async def extract_features_debug(file: UploadFile):
    """
    Debug endpoint that returns features + annotated image (base64 JPEG).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")
    contents = await file.read()
    import base64
    import numpy as np
    import cv2
    from vision_pipeline.feature_extractor import FeatureExtractor

    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    extractor = FeatureExtractor()
    features, annotated = extractor.extract_and_draw(frame)

    _, jpg = cv2.imencode(".jpg", annotated)
    image_b64 = base64.b64encode(jpg.tobytes()).decode()

    return {"features": features.tolist(), "dim": int(len(features)), "image_b64": image_b64}


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
    if not file.content_type or not file.content_type.startswith("audio/"):
        if file.content_type != "application/octet-stream":
            raise HTTPException(status_code=400, detail="Expected an audio upload")

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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")
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
    # Optional token check (query param: ?token=...)
    token = ws.query_params.get("token")
    if STREAM_TOKEN_REQUIRED and STREAM_TOKEN and token != STREAM_TOKEN:
        await ws.close(code=1008)
        return

    await ws.accept()
    ws_clients.append(ws)
    fps_limit = 15
    window = 8
    min_confidence = 0.4
    max_frame_bytes = 1_000_000
    use_sequence = False
    sequence_window = 30
    from gesture_recognition.utils.temporal_filter import TemporalFilter, TemporalSequenceBuffer
    smoother = TemporalFilter(window=window)
    sequence_buffer = TemporalSequenceBuffer(window=sequence_window)
    last_frame_time = 0.0
    dropped_frames = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                image_bytes = msg["bytes"]
                # Throttle frame rate
                now = time.time()
                if fps_limit > 0 and (now - last_frame_time) < (1 / fps_limit):
                    dropped_frames += 1
                    continue
                last_frame_time = now
                # Guardrail on payload size (max 1MB)
                if len(image_bytes) > max_frame_bytes:
                    await ws.send_json({"error": "frame_too_large", "detail": f"max {max_frame_bytes} bytes"})
                    continue
                try:
                    features = realtime.extract_features_from_bytes(image_bytes)
                    start = time.time()
                    result = realtime.classify_gesture(features.tolist())
                    latency_ms = int((time.time() - start) * 1000)
                    result["frame_id"] = int(time.time() * 1000)
                    result["latency_ms"] = latency_ms
                    result["server_ts"] = int(time.time() * 1000)
                    result["dropped_frames"] = dropped_frames
                except Exception as e:
                    await ws.send_json({"error": "frame_decode_failed", "detail": str(e)})
                    continue
                conf = result.get("confidence")
                if conf is not None and conf < min_confidence:
                    result["gesture_id"] = None
                await ws.send_json(result)
                continue
            else:
                data = msg.get("json") or {}

            # Optional config message:
            # {"type": "config", "fps": 15, "window": 8, "min_confidence": 0.4}
            if data.get("type") == "config":
                payload = StreamInitRequest(
                    fps=data.get("fps", fps_limit),
                    window=data.get("window", window),
                    min_confidence=data.get("min_confidence", min_confidence),
                    max_frame_bytes=data.get("max_frame_bytes", max_frame_bytes),
                    use_sequence=data.get("use_sequence", use_sequence),
                    sequence_window=data.get("sequence_window", sequence_window),
                )
                fps_limit = int(payload.fps)
                window = int(payload.window)
                min_confidence = float(payload.min_confidence)
                max_frame_bytes = int(payload.max_frame_bytes)
                use_sequence = bool(payload.use_sequence)
                sequence_window = int(payload.sequence_window)
                smoother = TemporalFilter(window=window)
                sequence_buffer = TemporalSequenceBuffer(window=sequence_window)
                await ws.send_json({
                    "type": "config_ack",
                    "fps": fps_limit,
                    "window": window,
                    "min_confidence": min_confidence,
                    "max_frame_bytes": max_frame_bytes,
                    "use_sequence": use_sequence,
                    "sequence_window": sequence_window,
                })
                continue

            # Optional: accept base64-encoded image frames
            if data.get("type") == "frame" and data.get("image_b64"):
                import base64
                try:
                    image_bytes = base64.b64decode(data.get("image_b64"))
                    if len(image_bytes) > max_frame_bytes:
                        await ws.send_json({"error": "frame_too_large", "detail": f"max {max_frame_bytes} bytes"})
                        continue
                    features = realtime.extract_features_from_bytes(image_bytes)
                    result = realtime.classify_gesture(features.tolist())
                    result["frame_id"] = data.get("frame_id", 0)
                    result["server_ts"] = int(time.time() * 1000)
                except Exception as e:
                    await ws.send_json({"error": "frame_decode_failed", "detail": str(e)})
                    continue
            else:
                if use_sequence:
                    keypoints = data.get("keypoints", [])
                    sequence = sequence_buffer.update(keypoints)
                    data["use_sequence"] = True
                    data["sequence_ready"] = sequence is not None
                    data["sequence_length"] = len(sequence) if sequence is not None else 0
                result = realtime.process_stream_frame(data)
            gesture_id = result.get("gesture_id")
            conf = result.get("confidence")
            if conf is not None and conf < min_confidence:
                gesture_id = None
                result["gesture_id"] = None
            smooth_id = smoother.update(gesture_id)
            if smooth_id is not None:
                result["gesture_id"] = smooth_id
                label = loader.gesture_label(smooth_id)
                if label:
                    result["gesture_label"] = label
            result["dropped_frames"] = dropped_frames
            result["server_ts"] = int(time.time() * 1000)
            await ws.send_json(result)
    except WebSocketDisconnect:
        ws_clients.remove(ws)


if __name__ == "__main__":
    reload = os.getenv("UVICORN_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("api_server.server:app", host="0.0.0.0", port=8000, reload=reload)
