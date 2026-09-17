"""
Backend smoke test - manual verification of inference pipeline.

Run this after starting the backend to verify:
1. Health endpoint works
2. System status shows model loaded
3. A real test image can be inspected
4. Results are properly generated
"""

import sys
import time
import json
from pathlib import Path
import requests
from PIL import Image, ImageDraw
import io

BASE_URL = "http://localhost:8000/api/v1"
PROJECT_ROOT = Path(r"D:\Projects\AeroInspectAI")


def create_test_image() -> bytes:
    """Create a simple test image."""
    # Create a test image with some contrast
    img = Image.new("RGB", (640, 480), color="white")
    draw = ImageDraw.Draw(img)
    
    # Draw some lines to simulate cracks
    draw.line([(100, 100), (500, 300)], fill="black", width=3)
    draw.line([(150, 150), (450, 400)], fill="black", width=2)
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


def test_health():
    """Test health endpoint."""
    print("\n[TEST] Health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"  Status: {data.get('status')}")
        print(f"  Service: {data.get('service')}")
        print(f"  Version: {data.get('version')}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_system_status():
    """Test system status endpoint."""
    print("\n[TEST] System status...")
    try:
        response = requests.get(f"{BASE_URL}/system/status")
        response.raise_for_status()
        data = response.json()
        
        # Check model
        model_loaded = data.get("model", {}).get("loaded", False)
        print(f"  Model loaded: {model_loaded}")
        
        if not model_loaded:
            print("  ERROR: Model not loaded!")
            return False
        
        # Check compute
        device = data.get("compute", {}).get("device", "unknown")
        gpu_name = data.get("compute", {}).get("gpu_name", "N/A")
        cuda_available = data.get("compute", {}).get("cuda_available", False)
        
        print(f"  Device: {device}")
        print(f"  GPU: {gpu_name}")
        print(f"  CUDA available: {cuda_available}")
        
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_inspection_with_test_image():
    """Test inspection with a generated test image."""
    print("\n[TEST] Inspection with test image...")
    try:
        # Create test image
        test_image = create_test_image()
        
        files = {"file": ("test_image.jpg", test_image)}
        params = {
            "confidence": 0.25,
            "iou": 0.70
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/inspect",
            files=files,
            data=params
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        response.raise_for_status()
        result = response.json()
        
        inspection_id = result.get("inspection_id")
        status = result.get("status")
        detections = result.get("detections", [])
        summary = result.get("summary", {})
        
        print(f"  Inspection ID: {inspection_id}")
        print(f"  Status: {status}")
        print(f"  Detections: {len(detections)}")
        print(f"  Average confidence: {summary.get('average_confidence', 'N/A'):.3f}")
        print(f"  Max confidence: {summary.get('max_confidence', 'N/A'):.3f}")
        print(f"  Inference time: {result.get('inference', {}).get('inference_time_ms', 'N/A'):.1f}ms")
        print(f"  Total time: {elapsed_ms:.1f}ms")
        
        return inspection_id
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_retrieve_results(inspection_id: str):
    """Test retrieving inspection results."""
    print(f"\n[TEST] Retrieving results for {inspection_id}...")
    try:
        # Get result JSON
        response = requests.get(f"{BASE_URL}/results/{inspection_id}")
        response.raise_for_status()
        result = response.json()
        print(f"  Result JSON retrieved: OK")
        
        # Get original image
        response = requests.get(f"{BASE_URL}/results/{inspection_id}/original")
        response.raise_for_status()
        print(f"  Original image retrieved: {len(response.content)} bytes")
        
        # Get annotated image
        response = requests.get(f"{BASE_URL}/results/{inspection_id}/annotated")
        response.raise_for_status()
        print(f"  Annotated image retrieved: {len(response.content)} bytes")
        
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_inspections_history():
    """Test inspection history endpoint."""
    print("\n[TEST] Inspection history...")
    try:
        response = requests.get(f"{BASE_URL}/inspections")
        response.raise_for_status()
        history = response.json()
        print(f"  Retrieved history: {len(history)} records")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    """Run smoke tests."""
    print("=" * 60)
    print("AEROINSPECT AI — BACKEND SMOKE TEST")
    print("=" * 60)
    
    # Wait for backend to be ready
    print("\nWaiting for backend to be ready...")
    for attempt in range(30):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print("Backend is ready!")
                break
        except Exception:
            if attempt < 29:
                time.sleep(1)
    else:
        print("ERROR: Backend not responding!")
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Health
    tests_total += 1
    if test_health():
        tests_passed += 1
    
    # Test 2: System status
    tests_total += 1
    if test_system_status():
        tests_passed += 1
    
    # Test 3: Inspection
    tests_total += 1
    inspection_id = test_inspection_with_test_image()
    if inspection_id:
        tests_passed += 1
        
        # Test 4: Results retrieval
        tests_total += 1
        if test_retrieve_results(inspection_id):
            tests_passed += 1
    
    # Test 5: History
    tests_total += 1
    if test_inspections_history():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\n✓ ALL TESTS PASSED - Backend is fully functional")
        return 0
    else:
        print(f"\n✗ TESTS FAILED - {tests_total - tests_passed} failures")
        return 1


if __name__ == "__main__":
    sys.exit(main())
