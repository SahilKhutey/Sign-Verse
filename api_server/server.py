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
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
from typing import Dict, List, Optional
import uvicorn
import os
import shutil
import time
import tempfile
import asyncio
import cv2
import numpy as np
import psutil
from dotenv import load_dotenv

from api_server.model_loader import ModelLoader
from api_server.realtime_inference import RealtimeInference
from api_server.routers import auth, translation, sign_recognition, sign_generation, feedback
from api_server.core.security import verify_token
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
# from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# ── Foundation Model Configuration (SFM-v2) ──────────────────────────────────
# Replaced legacy vLLM initialization with OptimizedFoundationTransformer
# to prevent startup hangs and allow 30+ FPS on standard hardware.
loader = ModelLoader()
realtime = RealtimeInference(loader)

app = FastAPI(
    title="SignVerse AI Foundation Server",
    version="2.5.0",
    description="Holistic Sign Language AI — 3,258-dim Foundation Inference"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(auth.router)
app.include_router(translation.router)
app.include_router(sign_recognition.router)
app.include_router(sign_generation.router)
app.include_router(feedback.router)

# ── Hybrid Edge/Cloud Routing ──────────────────────────────────────────────────
try:
    from hybrid.cloud_inference_ws import router as hybrid_router
    app.include_router(hybrid_router)
    import asyncio, hybrid.load_balancer as _lb_mod
    @app.on_event("startup")
    async def _start_health_checks():
        from hybrid.cloud_inference_ws import _load_balancer
        asyncio.create_task(_load_balancer.health_check_loop())
except ImportError as _e:
    import logging; logging.getLogger("server").warning(f"Hybrid router not loaded: {_e}")

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
_fingerspell_sessions: Dict[str, object] = {}

# Speech-to-sign pipeline (same behavior as api_gateway)
_speech_pipeline = None
os.makedirs("temp", exist_ok=True)


def _get_speech_pipeline():
    global _speech_pipeline
    if _speech_pipeline is None:
        from ai_engine.inference_pipeline import InferencePipeline
        _speech_pipeline = InferencePipeline()
    return _speech_pipeline


def _get_fingerspell_decoder(session_id: str, min_confidence: float):
    from ai_models.gesture_recognition.fingerspelling import FingerSpellingDecoder

    decoder = _fingerspell_sessions.get(session_id)
    if decoder is None:
        # Keep memory bounded for long-running API servers.
        if len(_fingerspell_sessions) >= 200:
            oldest = next(iter(_fingerspell_sessions))
            _fingerspell_sessions.pop(oldest, None)
        decoder = FingerSpellingDecoder(min_confidence=min_confidence)
        _fingerspell_sessions[session_id] = decoder
    else:
        try:
            decoder.min_confidence = float(min_confidence)
        except Exception:
            pass
    return decoder


class KeypointRequest(BaseModel):
    keypoints: conlist(float, min_length=1) = Field(..., description="Flattened keypoint vector")


class KeypointSequenceRequest(BaseModel):
    sequence: conlist(List[float], min_length=1) = Field(..., description="Sequence of keypoint vectors")


class SignToTextRequest(BaseModel):
    sequence: conlist(List[float], min_length=1) = Field(..., description="Sequence of keypoint frames")


class TextToSignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="ASL")


class TextToSignVideoPlanRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="ASL")
    dictionary_manifest: str = Field(
        default=os.path.join("datasets", "sign_dictionary", "manifest.csv"),
        description="CSV or JSON dictionary manifest for concatenative synthesis",
    )
    strict_manifest: bool = Field(
        default=False,
        description="When true, return error if dictionary manifest is missing.",
    )


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


@app.post("/gesture/classify-sequence")
async def classify_gesture_sequence(data: KeypointSequenceRequest):
    """Classify gesture from a temporal keypoint sequence."""
    sequence = data.sequence
    result = realtime.classify_gesture_sequence(sequence)
    gesture_id = result.get("gesture_id")
    label = loader.gesture_label(gesture_id)
    if label:
        result["gesture_label"] = label
    result["sequence_length"] = len(sequence)
    return result


@app.post("/gesture/classify-image-cnn")
async def classify_gesture_image_cnn(file: UploadFile, min_confidence: float = 0.4):
    """
    Classify ASL letter/gesture from image using optional Keras CNN model.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")
    try:
        contents = await file.read()
        import numpy as np
        import cv2

        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")

        model = loader.get_asl_cnn_model()
        pred = model.predict(frame)
        conf = pred.get("confidence")
        if conf is not None and float(conf) < float(min_confidence):
            pred["label"] = None
        return pred
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASL CNN inference failed: {exc}")


@app.post("/gesture/classify-image-cnn-fingerspell")
async def classify_gesture_image_cnn_fingerspell(
    file: UploadFile,
    session_id: str = "default",
    min_confidence: float = 0.4,
    reset: bool = False,
):
    """
    ASL alphabet inference + session-based finger-spelling decoding.
    """
    if not session_id or len(session_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")

    if reset:
        _fingerspell_sessions.pop(session_id, None)

    try:
        contents = await file.read()
        import numpy as np
        import cv2

        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")

        model = loader.get_asl_cnn_model()
        pred = model.predict(frame)
        conf = pred.get("confidence")
        if conf is not None and float(conf) < float(min_confidence):
            pred["label"] = None

        decoder = _get_fingerspell_decoder(session_id=session_id, min_confidence=min_confidence)
        decoded = decoder.update(label=pred.get("label"), confidence=conf)
        return {
            **pred,
            "session_id": session_id,
            "stable_label": decoded.get("stable_label"),
            "committed": decoded.get("committed"),
            "text": decoded.get("text"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASL fingerspelling inference failed: {exc}")


@app.post("/gesture/classify-video-lstm")
async def classify_gesture_video_lstm(file: UploadFile, min_confidence: float = 0.4):
    """
    Classify isolated sign from uploaded video using optional CNN+LSTM model.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        if file.content_type != "application/octet-stream":
            raise HTTPException(status_code=400, detail="Expected a video upload")
    try:
        video_bytes = await file.read()
        if not video_bytes:
            raise HTTPException(status_code=400, detail="Empty video file")
        model = loader.get_video_lstm_model()
        pred = model.predict_video_bytes(video_bytes)
        conf = pred.get("confidence")
        if conf is not None and float(conf) < float(min_confidence):
            pred["label"] = None
        return pred
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video LSTM inference failed: {exc}")


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


@app.post("/translate/video-to-text")
async def video_to_text(
    file: UploadFile,
    sample_every: int = 2,
    max_frames: int = 90,
):
    """
    Translate an uploaded sign video clip to text.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        if file.content_type != "application/octet-stream":
            raise HTTPException(status_code=400, detail="Expected a video upload")
    if sample_every < 1 or sample_every > 30:
        raise HTTPException(status_code=400, detail="sample_every must be between 1 and 30")
    if max_frames < 5 or max_frames > 400:
        raise HTTPException(status_code=400, detail="max_frames must be between 5 and 400")

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file")

    tmp_path = None
    try:
        from vision_pipeline.video_embedding import VideoEmbeddingExtractor

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="temp") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        extractor = VideoEmbeddingExtractor(
            sample_every=sample_every,
            max_frames=max_frames,
            allow_mock=False,
        )
        frame_features = extractor.extract_frame_features(tmp_path)
        if frame_features.ndim != 2 or frame_features.shape[0] == 0:
            raise HTTPException(status_code=400, detail="No decodable sign frames found in video")

        text = realtime.video_to_text(frame_features.tolist())
        return {
            "text": text,
            "num_frames": int(frame_features.shape[0]),
            "feature_dim": int(frame_features.shape[1]),
            "sample_every": int(sample_every),
            "max_frames": int(max_frames),
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        detail = str(exc)
        if "mock mode" in detail.lower():
            raise HTTPException(status_code=503, detail=detail)
        raise HTTPException(status_code=500, detail=f"Video-to-text inference failed: {detail}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video-to-text inference failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


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
    try:
        text = data.text
        tokens = realtime.text_to_sign(text, language=data.language)
        return {"tokens": tokens}
    except Exception as e:
        print(f"ERROR in text_to_sign: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/text-to-sign-video-plan")
async def text_to_sign_video_plan(data: TextToSignVideoPlanRequest):
    """
    Build concatenative clip plan for sentence-level sign video synthesis.
    """
    try:
        from nlp_translation.concatenative_synthesis import ConcatenativeSynthesis

        tokens = realtime.text_to_sign(data.text, language=data.language)
        if isinstance(tokens, str):
            tokens = [t for t in tokens.split() if t.strip()]
        if not isinstance(tokens, list):
            tokens = []

        synth = ConcatenativeSynthesis(
            dictionary_manifest=data.dictionary_manifest,
            strict_exists=data.strict_manifest,
        )
        planned = synth.build_plan(tokens)
        return {
            "text": data.text,
            "tokens": tokens,
            "dictionary_manifest": data.dictionary_manifest,
            **planned,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Concatenative planning failed: {exc}")


# MJPEG Video Feed Generator
def gen_video_frames():
    import cv2
    import numpy as np
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera not found, using color bars for fallback.")
        while True:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera Offline - SignVerse", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret: continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)

    extractor = loader.get_feature_extractor()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process for overlays
            _, annotated = extractor.extract_and_draw(frame)
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret: continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video-feed")
async def video_feed():
    """MJPEG streaming endpoint for the dashboard video player."""
    return StreamingResponse(gen_video_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# ... existing code ...

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Real-time Gesture Streaming Engine (SFM-v2).
    
    Optimized for 30+ FPS and 543 MultiPipe Holistic landmarks.
    """
    await ws.accept()
    ws_clients.append(ws)
    await ws.send_json({"type": "connected", "status": "ok"})
    
    fps_limit = 30 
    min_confidence = 0.5
    sequence_window = 30
    
    from gesture_recognition.utils.temporal_filter import TemporalFilter, TemporalSequenceBuffer
    smoother = TemporalFilter(window=12)
    sequence_buffer = TemporalSequenceBuffer(window=sequence_window)
    
    last_frame_time = 0.0
    dropped_frames = 0
    
    process = psutil.Process(os.getpid())
    
    async def telemetry_pusher():
        try:
            while True:
                mem = process.memory_info().rss
                cpu = psutil.cpu_percent()
                await ws.send_json({
                    "type": "telemetry",
                    "cpu_usage": cpu,
                    "memory_usage": mem,
                    "server_ts": int(time.time() * 1000),
                    "status": "active"
                })
                await asyncio.sleep(1)
        except Exception:
            pass

    telemetry_task = asyncio.create_task(telemetry_pusher())
    
    try:
        while True:
            # We use wait_for to allow the loop to be interrupted by the telemetry task if needed
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                image_bytes = msg["bytes"]
                
                now = time.time()
                if fps_limit > 0 and (now - last_frame_time) < (1 / fps_limit):
                    dropped_frames += 1
                    continue
                last_frame_time = now
                
                try:
                    features = realtime.extract_features_from_bytes(image_bytes)
                    start_inf = time.time()
                    
                    sequence = sequence_buffer.update(features.tolist())
                    if sequence is not None:
                        result = realtime.classify_gesture_sequence(sequence)
                        result["sequence_ready"] = True
                    else:
                        result = realtime.classify_gesture_sequence([features.tolist()])
                        result["sequence_ready"] = False
                    
                    latency_ms = int((time.time() - start_inf) * 1000)
                    result["latency_ms"] = latency_ms
                    result["server_ts"] = int(time.time() * 1000)
                    result["dropped_frames"] = dropped_frames
                    
                    # ── Added: Real-time System Telemetry ────────────────────────
                    result["cpu_usage"] = psutil.cpu_percent()
                    result["memory_usage"] = process.memory_info().rss # bytes
                    result["fps"] = 1.0 / (time.time() - now + 1e-6) if last_frame_time > 0 else 0
                    
                except Exception as e:
                    await ws.send_json({"error": "processing_failed", "detail": str(e)})
                    continue

                gesture_id = result.get("gesture_id")
                if result.get("confidence", 0) < min_confidence:
                    gesture_id = None
                
                smooth_id = smoother.update(gesture_id)
                if smooth_id is not None:
                    result["gesture_id"] = smooth_id
                    label = loader.gesture_label(smooth_id)
                    result["gesture_label"] = label or "unknown"
                
                await ws.send_json(result)
                
            elif msg.get("json") is not None:
                data = msg["json"]
                if data.get("type") == "config":
                    fps_limit = int(data.get("fps", fps_limit))
                    min_confidence = float(data.get("min_confidence", min_confidence))
                    await ws.send_json({"type": "config_ack", "fps": fps_limit, "conf": min_confidence})

    except WebSocketDisconnect:
        if ws in ws_clients: ws_clients.remove(ws)
    except Exception as e:
        if ws in ws_clients: ws_clients.remove(ws)
    finally:
        telemetry_task.cancel()

if __name__ == "__main__":
    reload = os.getenv("UVICORN_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("api_server.server:app", host="0.0.0.0", port=8000, reload=reload)
