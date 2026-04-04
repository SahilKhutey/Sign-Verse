"""
Main FastAPI application for SignVerse System.
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from loguru import logger
import uuid
import time

from configs import get_config
from core.data_manager import data_manager
from models.inference.predict_pose import get_pose_predictor
from simulation.blender import blender_manager

# Import routers
from .routes.upload import router as upload_router
from .routes.inference import router as inference_router
from .routes.simulation import router as simulation_router
from .routes.monitoring import router as monitoring_router
from .routes.youtube import router as youtube_router
from .routes.datasets import router as datasets_router
from .routes.dataset import router as dataset_router

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting SignVerse API server...")
    
    # Initialize components
    config = get_config()
    
    # Create necessary directories
    data_manager._ensure_directories()
    
    # Load ML models
    try:
        pose_predictor = get_pose_predictor()
        if pose_predictor:
            logger.success("Pose prediction model loaded successfully")
        else:
            logger.warning("Pose predictor not initialized.")
    except Exception as e:
        logger.error(f"Failed to load pose model: {e}")
    
    # Initialize Blender connection
    try:
        if blender_manager:
            blender_manager.setup_scene()
            logger.success("Blender integration initialized")
        else:
            logger.warning("Blender manager not available")
    except Exception as e:
        logger.warning(f"Blender not available: {e}")
    
    startup_time = time.time()
    app.state.startup_time = startup_time
    app.state.request_count = 0
    
    logger.success(f"API server started in {time.time() - startup_time:.2f}s")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SignVerse API server...")
    # Cleanup resources here

# Create FastAPI application
app = FastAPI(
    title="SignVerse API",
    description="REST API for SignVerse Pose Estimation and Simulation System",
    version="1.0.0",
    docs_url=None,  # We'll customize the docs URL
    redoc_url=None,
    lifespan=lifespan
)

# CORS middleware
try:
    cors_origins = get_config().app.cors_origins
except Exception:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom request middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add request processing time and ID to headers."""
    request_id = str(uuid.uuid4())
    request.state.id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-SignVerse-Version"] = get_config().app.version
    
    # Update request counter
    app.state.request_count += 1
    
    logger.info(f"Request {request_id} completed in {process_time:.3f}s")
    return response

# Include routers
app.include_router(upload_router, prefix="/api/v1", tags=["Upload"])
app.include_router(inference_router, prefix="/api/v1", tags=["Inference"])
app.include_router(simulation_router, prefix="/api/v1", tags=["Simulation"])
app.include_router(monitoring_router, prefix="/api/v1", tags=["Monitoring"])
app.include_router(youtube_router, prefix="/api/v1/youtube", tags=["YouTube Analytics"])
app.include_router(datasets_router, prefix="/api/v1/datasets", tags=["Dataset Management"])
app.include_router(dataset_router, prefix="/api/v1/dataset", tags=["Dataset Quick Actions"])

# Custom docs endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": get_config().app.version,
        "uptime": time.time() - getattr(app.state, 'startup_time', time.time()),
        "requests_processed": getattr(app.state, 'request_count', 0)
    }

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Welcome to SignVerse API",
        "version": get_config().app.version,
        "docs": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        app,
        host=config.app.api_host,
        port=config.app.api_port,
        workers=config.app.api_workers
    )
