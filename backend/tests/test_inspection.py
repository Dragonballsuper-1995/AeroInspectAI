"""Tests for inspection endpoint."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_inspect_invalid_file():
    """Test inspection with invalid file."""
    response = client.post("/api/v1/inspect", data={})
    assert response.status_code == 400


def test_inspect_empty_file():
    """Test inspection with empty file."""
    files = {"file": ("empty.jpg", b"")}
    response = client.post("/api/v1/inspect", files=files)
    assert response.status_code == 400


def test_inspect_invalid_image_format():
    """Test inspection with invalid image format."""
    files = {"file": ("test.txt", b"not an image")}
    response = client.post("/api/v1/inspect", files=files)
    assert response.status_code == 400


def test_inspect_invalid_confidence():
    """Test inspection with invalid confidence parameter."""
    # Create a valid test image (1x1 pixel)
    from PIL import Image
    import io
    
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    files = {"file": ("test.jpg", img_bytes.getvalue())}
    response = client.post(
        "/api/v1/inspect",
        files=files,
        data={"confidence": 1.5}
    )
    assert response.status_code == 422


def test_inspect_invalid_iou():
    """Test inspection with invalid IoU parameter."""
    from PIL import Image
    import io
    
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    files = {"file": ("test.jpg", img_bytes.getvalue())}
    response = client.post(
        "/api/v1/inspect",
        files=files,
        data={"iou": 1.5}
    )
    assert response.status_code == 422


def test_inspections_history():
    """Test inspections history endpoint."""
    response = client.get("/api/v1/inspections")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
