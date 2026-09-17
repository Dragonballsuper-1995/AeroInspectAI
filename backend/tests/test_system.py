"""Tests for system endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_system_status():
    """Test system status endpoint."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    
    data = response.json()
    
    # Check backend status
    assert "backend" in data
    assert data["backend"]["status"] == "running"
    
    # Check model status
    assert "model" in data
    assert data["model"]["loaded"] == True
    assert data["model"]["name"] == "YOLOv8n-Seg"
    assert "classes" in data["model"]
    
    # Check compute status
    assert "compute" in data
    assert "device" in data["compute"]
    assert "cuda_available" in data["compute"]
    assert "torch_version" in data["compute"]
    assert "ultralytics_version" in data["compute"]
