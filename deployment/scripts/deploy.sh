#!/bin/bash
# SignVerse Deployment Script

set -e

echo "Building Docker images..."
docker build -f deployment/docker/Dockerfile.ai -t signverse/ai-engine:latest .
docker build -f deployment/docker/Dockerfile.backend -t signverse/backend:latest .

echo "Pushing images..."
docker push signverse/ai-engine:latest
docker push signverse/backend:latest

echo "Deploying to Kubernetes..."
kubectl apply -f deployment/kubernetes/ai_engine.yaml
kubectl apply -f deployment/kubernetes/backend.yaml

echo "Waiting for rollout..."
kubectl rollout status deployment/signverse-ai-engine
kubectl rollout status deployment/signverse-backend

echo "Deployment complete!"
kubectl get pods -l app=signverse-ai
kubectl get pods -l app=signverse-backend
