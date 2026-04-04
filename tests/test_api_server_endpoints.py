"""
Integration tests for the SignVerse API Server endpoints.
Mocks heavy AI model loading and external dependencies to ensure rapid test execution.
"""
import pytest
import sys
from unittest.mock import Mock, patch

# CRITICAL: Mock heavy dependencies in sys.modules BEFORE any api_server imports
# This prevents the server from attempting to load GPU-bound or missing libraries
for module in ['vllm', 'mediapipe', 'cv2', 'torch']:
    sys.modules[module] = Mock()

# Mock the specific classes that trigger library initialization
sys.modules['api_server.model_loader'] = Mock()
sys.modules['api_server.realtime_inference'] = Mock()

from api_server.server import app
from fastapi.testclient import TestClient

class TestAPIServer:
    """Test API endpoints for SignVerse AI services."""
    
    @pytest.fixture
    def client(self):
        """Standard FastAPI test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Verify server health status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_vision_extract_invalid_file(self, client):
        """Test vision extraction with invalid file type."""
        response = client.post("/vision/extract", files={"file": ("test.txt", b"hello", "text/plain")})
        assert response.status_code == 400

    def test_gesture_classify_empty(self, client):
        """Test gesture classification with empty payload."""
        response = client.post("/gesture/classify", json={})
        # FastAPI returns 422 for pydantic validation errors
        assert response.status_code == 422
