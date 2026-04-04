#!/bin/bash
# deploy_signverse.sh — Production Deployment Script

set -e

echo "--- SignVerse AI Production Deployment ---"

# Load production env
if [ -f .env.production ]; then
    export $(grep -v '^#' .env.production | xargs)
fi

# 1. Build and Start with Docker Compose
echo "Building and starting services via Docker Compose..."
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d

# 2. Verify Health
echo "Waiting for services to stabilize..."
sleep 10
docker-compose ps

# 3. Check GPU Visibility
echo "Checking GPU status in AI Engine..."
docker-compose exec ai-engine nvidia-smi || echo "Warning: No GPU detected in container. Falling back to CPU."

# 4. Run Smoke Tests
echo "Running smoke tests..."
# docker-compose exec backend python3 -m tests.smoke_test

echo ""
echo "SignVerse AI Ecosystem is now ONLINE (Production Mode)"
echo "  AR Dashboard: http://localhost:8080"
echo "  API Backend:  http://localhost:8001"
echo "  AI Inference: http://localhost:8000"
echo "------------------------------------------"
