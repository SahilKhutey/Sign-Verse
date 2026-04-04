#!/bin/bash

# SignVerse Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Environment variables
ENV=${1:-production}
REGISTRY=${2:-your-registry}
TAG=${3:-latest}

echo -e "${GREEN}Deploying SignVerse (${ENV})...${NC}"

# Build and push Docker images
build_and_push() {
    local service=$1
    local dockerfile=$2
    
    echo -e "${YELLOW}Building ${service}...${NC}"
    docker build -f ${dockerfile} -t ${REGISTRY}/signverse-${service}:${TAG} .
    
    echo -e "${YELLOW}Pushing ${service}...${NC}"
    docker push ${REGISTRY}/signverse-${service}:${TAG}
}

# Build all services
build_and_push "backend" "infra/docker/backend.Dockerfile"
build_and_push "frontend" "infra/docker/frontend.Dockerfile"

# Update Kubernetes deployments
echo -e "${YELLOW}Updating Kubernetes deployments...${NC}"
kubectl set image deployment/signverse-backend backend=${REGISTRY}/signverse-backend:${TAG} -n signverse
kubectl set image deployment/signverse-frontend frontend=${REGISTRY}/signverse-frontend:${TAG} -n signverse

# Wait for rollout
echo -e "${YELLOW}Waiting for rollout...${NC}"
kubectl rollout status deployment/signverse-backend -n signverse --timeout=300s
kubectl rollout status deployment/signverse-frontend -n signverse --timeout=300s

echo -e "${GREEN}Deployment completed successfully!${NC}"
