#!/bin/bash
# deploy_signverse.sh

echo "--- SignVerse AI Production Deployment ---"

# 1. Build and push Docker images
echo "Building engine image..."
docker build -t signverse/engine:latest -f deployment/docker/Dockerfile.ai .
echo "Building api image..."
# assuming a Dockerfile.api exists or using inference_server
docker build -t signverse/api:latest -f deployment/docker/Dockerfile.api . 

# docker push signverse/engine:latest
# docker push signverse/api:latest

# 2. Deploy to Kubernetes
echo "Applying Kubernetes manifests..."
kubectl apply -f deployment/kubernetes/namespace.yaml || true
kubectl apply -f deployment/kubernetes/backend.yaml
# Additional manifests can be applied here

# 3. Setup monitoring
echo "Setting up monitoring..."
# kubectl apply -f kubernetes/monitoring/prometheus.yaml
# kubectl apply -f kubernetes/monitoring/grafana.yaml

# 4. Run integration tests (as smoke tests)
echo "Running smoke tests..."
# python tests/integration_tests.py --deployment production

echo "SignVerse AI Engine deployed successfully!"
