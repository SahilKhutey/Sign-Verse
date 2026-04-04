#!/bin/bash

# Monitoring script for SignVerse
set -e

echo "=== SignVerse Monitoring ==="
echo "Timestamp: $(date)"

# Check Kubernetes pods
echo -e "\n=== Kubernetes Pods ==="
kubectl get pods -n signverse

# Check services
echo -e "\n=== Services ==="
kubectl get svc -n signverse

# Check ingress
echo -e "\n=== Ingress ==="
kubectl get ingress -n signverse

# Check resource usage
echo -e "\n=== Resource Usage ==="
kubectl top pods -n signverse

# Check logs (last 10 lines of backend)
echo -e "\n=== Backend Logs ==="
kubectl logs deployment/signverse-backend -n signverse --tail=10

# Health check
echo -e "\n=== Health Check ==="
curl -s http://localhost:8000/health | jq .

echo -e "\nMonitoring completed!"
