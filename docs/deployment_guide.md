# Deployment Guide

This repo supports local multi-service runs and containerized deployments.

Important: ensure Python deps are installed and imports resolve in the target environment (Docker image or local venv).

## Local (Scripts)

- Start services: `deployment/scripts/start_services.sh`
  - Starts:
    - AI Engine (FastAPI) on `:8000` via `uvicorn api_gateway.main:app`
    - Backend on `:8001` via `uvicorn backend.app:app`
    - Avatar API on `:8002` via `uvicorn avatar_animation.unity_bridge.animation_api:app`

Notes:
- The script assumes a POSIX shell and `venv/bin/activate`. On Windows, run equivalent commands in PowerShell, or use Docker Compose.

## Docker Compose

- File: `docker-compose.yml`
  - Services:
    - `ai-engine` (port 8000, optional NVIDIA GPU)
    - `backend` (port 8001)
    - `avatar-api` (port 8002)

Run:
- `docker compose up --build`

## Dockerfiles

- AI image: `deployment/docker/Dockerfile.ai`
  - Installs `requirements.txt`
  - Copies: `ai_engine/`, `ai_models/`, `api_gateway/`, `api_server/`, `gesture_recognition/`, `models/`, `nlp_translation/`, `vision_pipeline/`, `avatar_animation/`
  - Starts: `uvicorn api_gateway.main:app`
- Backend image: `deployment/docker/Dockerfile.backend`

## Kubernetes

- AI Engine: `deployment/kubernetes/ai_engine.yaml`
- Backend: `deployment/kubernetes/backend.yaml`
- Script: `deployment/scripts/deploy.sh` (build, push, apply, rollout)

## Recommended Pre-Deploy Checklist

- Verify all imports resolve inside the container (no dash/underscore mismatches).
- Confirm model artifact paths exist under `models/` and are non-empty.
- Add health checks and resource limits per environment (CPU/GPU).
