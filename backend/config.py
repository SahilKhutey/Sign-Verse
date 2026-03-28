"""
Backend Configuration
"""

import os


def _parse_origins(value: str):
    if not value:
        return []
    return [o.strip() for o in value.split(",") if o.strip()]

# Environment Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///signverse.db")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "signverse-secret-key-change-in-production")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# CORS
ALLOWED_ORIGINS = _parse_origins(os.getenv("ALLOWED_ORIGINS", "*"))

# Service URLs
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8000")
AVATAR_API_URL = os.getenv("AVATAR_API_URL", "http://localhost:8002")
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000")

# Enforce secrets in production
if ENVIRONMENT == "production":
    if SECRET_KEY == "signverse-secret-key-change-in-production":
        raise RuntimeError("SECRET_KEY must be set in production")
    if "*" in ALLOWED_ORIGINS:
        raise RuntimeError("ALLOWED_ORIGINS must be explicit in production")
