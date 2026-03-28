"""
SignVerse Backend Server

Main FastAPI application with CORS, routing, and lifespan management.
Stores user data, translation history, and dataset metadata.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import socketio
import asyncio
import os
import sys
import time
from dotenv import load_dotenv

# Load environment (optional .env)
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.logger import get_logger
from common.telemetry import setup_telemetry
from slowapi import _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = get_logger("signverse-backend")
from utils.rate_limit import limiter

from database.db_connection import init_db
from database.db_connection import SessionLocal
from sqlalchemy import text
from services.user_service import router as user_router
from services.translation_service import router as translation_router
from config import ALLOWED_ORIGINS

app = FastAPI(
    title="SignVerse Backend",
    version="1.0.0",
    description="Backend server for SignVerse sign language translation platform"
)

# Telemetry setup
setup_telemetry(app, service_name="signverse-backend")

# Rate Limiter setup
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Socket.io setup
socketio_origins = "*" if ("*" in ALLOWED_ORIGINS) else (ALLOWED_ORIGINS or [])
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins=socketio_origins)
sio_app = socketio.ASGIApp(sio, app)

# CORS
allow_all = "*" in ALLOWED_ORIGINS if ALLOWED_ORIGINS else False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else (ALLOWED_ORIGINS or []),
    allow_credentials=False if allow_all else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(user_router, prefix="/api/users", tags=["Users"])
app.include_router(translation_router, prefix="/api/translations", tags=["Translations"])

@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or os.urandom(8).hex()
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    response.headers["x-request-id"] = request_id
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
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "signverse-backend"}


@app.get("/health/ready")
async def readiness_check():
    db_ok = False
    inference_ok = False
    error = None
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        error = f"db_error: {e}"

    try:
        import httpx
        from config import INFERENCE_API_URL
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(INFERENCE_API_URL.rstrip("/") + "/health")
            inference_ok = resp.status_code == 200
            if not inference_ok:
                error = f"inference_error: {resp.text}"
    except Exception as e:
        error = f"inference_error: {e}"

    status = "ready" if db_ok and inference_ok else "degraded"
    return {
        "status": status,
        "db_ok": db_ok,
        "inference_ok": inference_ok,
        "error": error,
    }


@app.get("/dashboard")
async def dashboard():
    """
    Minimal HTML dashboard for MVP monitoring.
    """
    import json
    import os

    def _load_json(path, default):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return default
        return default

    model_manifest = _load_json("deployment/model_registry/manifest.json", {})
    nlp_eval = _load_json("reports/nlp_eval.json", {})

    html = f"""
    <html>
    <head>
      <title>SignVerse Dashboard</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        h1 {{ margin-bottom: 8px; }}
        .card {{ border: 1px solid #ddd; padding: 16px; margin-bottom: 16px; border-radius: 8px; }}
        pre {{ background: #f7f7f7; padding: 12px; border-radius: 6px; overflow-x: auto; }}
      </style>
    </head>
    <body>
      <h1>SignVerse MVP Dashboard</h1>
      <div class="card">
        <h3>Model Registry</h3>
        <pre>{json.dumps(model_manifest, indent=2)}</pre>
      </div>
      <div class="card">
        <h3>Latest NLP Eval</h3>
        <pre>{json.dumps(nlp_eval, indent=2)}</pre>
      </div>
    </body>
    </html>
    """
    return html


@app.on_event("startup")
async def startup():
    init_db()


# --- Socket.io Events ---
@sio.event
async def connect(sid, environ):
    logger.info("socket_connect", extra={"sid": sid})

@sio.event
async def disconnect(sid):
    logger.info("socket_disconnect", extra={"sid": sid})

@sio.event
async def join_room(sid, data):
    room = data.get("room")
    await sio.enter_room(sid, room)
    logger.info("socket_join_room", extra={"sid": sid, "room": room})
    await sio.emit("message", {"text": f"User joined {room}"}, room=room)

@sio.event
async def send_translation(sid, data):
    # Broadcast translation to everyone in the room
    room = data.get("room")
    translation = data.get("translation")
    await sio.emit("new_translation", translation, room=room)

if __name__ == "__main__":
    uvicorn.run(sio_app, host="0.0.0.0", port=8001)
