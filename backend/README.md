# AeroInspectAI Backend

Production-structured FastAPI backend for autonomous drone-based structural crack inspection. Performs real YOLOv8n-Seg instance segmentation inference on uploaded inspection images.

## Overview

```
Next.js Dashboard (localhost:3000)
        ↓
   HTTP / REST
        ↓
FastAPI Backend (localhost:8000)
        ↓
YOLOv8n-Seg Model
        ↓
Crack Detection + Segmentation
        ↓
JSON Results + Annotated Images
```

This backend:
- Loads the trained YOLOv8n-Seg model once at startup
- Performs real inference on uploaded images
- Returns segmentation results with polygon boundaries
- Generates annotated result images
- Maintains inspection history
- Provides RESTful API for the frontend

## Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA CUDA 11.8+ (optional, falls back to CPU)
- PyTorch 2.1+ with CUDA support
- Virtual environment

### Installation

1. **Create and activate virtual environment:**

   ```bash
   cd backend
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**

   ```bash
   # Copy example config
   copy .env.example .env
   
   # Edit .env if needed (model paths, device, etc.)
   ```

4. **Start the backend:**

   **Windows:**
   ```bash
   run_backend.bat
   ```

   **Linux/macOS:**
   ```bash
   bash run_backend.sh
   ```

   Or directly:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. **Verify it's running:**

   - Health check: http://localhost:8000/api/v1/health
   - API docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Configuration

Edit `.env` to customize:

```env
# Model configuration
AEROINSPECT_MODEL_PATH=D:\Projects\AeroInspectAI\experiments\exp001_yolov8n_seg\runs\baseline\weights\best.pt
AEROINSPECT_DATASET_YAML=D:\Projects\AeroInspectAI\datasets\aeroinspect_crack_v1\aeroinspect_crack_v1.yaml

# Device (0 for first GPU, -1 for CPU)
AEROINSPECT_DEVICE=0
AEROINSPECT_CUDA_ENABLED=True

# Inference defaults (overridable per request)
AEROINSPECT_CONFIDENCE=0.25
AEROINSPECT_IOU=0.70
AEROINSPECT_IMAGE_SIZE=640

# Storage
AEROINSPECT_UPLOAD_DIR=./storage/uploads
AEROINSPECT_INSPECTION_DIR=./storage/inspections

# Upload limit (MB)
AEROINSPECT_MAX_UPLOAD_MB=20

# Frontend CORS
AEROINSPECT_CORS_ORIGINS=["http://localhost:3000"]
```

## API Endpoints

### Health & System

**GET /api/v1/health**
```json
{
  "status": "healthy",
  "service": "AeroInspectAI Backend",
  "version": "0.1.0"
}
```

**GET /api/v1/system/status**
```json
{
  "backend": {
    "status": "running"
  },
  "model": {
    "loaded": true,
    "name": "YOLOv8n-Seg",
    "path": "...",
    "classes": {"0": "crack"}
  },
  "compute": {
    "device": "cuda:0",
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
    "torch_version": "2.1.1",
    "ultralytics_version": "8.0.215",
    "vram_gb": 6.0
  }
}
```

### Inspection

**POST /api/v1/inspect**

Upload image and run inference.

Request:
```
Content-Type: multipart/form-data

file: <image file>          [Required: JPG/PNG/WEBP]
confidence: 0.25            [Optional: 0.0-1.0]
iou: 0.70                   [Optional: 0.0-1.0]
imgsz: 640                  [Optional: 32-1920]
```

Response:
```json
{
  "inspection_id": "INS-20260916-000001",
  "status": "completed",
  "image": {
    "original_url": "/api/v1/results/INS-20260916-000001/original",
    "annotated_url": "/api/v1/results/INS-20260916-000001/annotated"
  },
  "summary": {
    "detections": 3,
    "cracks_detected": 3,
    "average_confidence": 0.87,
    "max_confidence": 0.94,
    "min_confidence": 0.81,
    "total_mask_area_pixels": 12345,
    "image_width": 1920,
    "image_height": 1080
  },
  "detections": [
    {
      "id": 1,
      "class_id": 0,
      "class_name": "crack",
      "confidence": 0.94,
      "bounding_box": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 500.8,
        "y2": 350.2
      },
      "mask": {
        "polygon": [
          [100, 210],
          [120, 215],
          [150, 230]
        ],
        "area_pixels": 4500,
        "area_ratio": 0.0022
      }
    }
  ],
  "inference": {
    "model": "YOLOv8n-Seg",
    "device": "cuda:0",
    "image_size": 640,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.70,
    "inference_time_ms": 35.2
  }
}
```

**GET /api/v1/inspections**

Get inspection history.

Query parameters:
- `limit`: Max records (default: 100)

Response:
```json
[
  {
    "inspection_id": "INS-20260916-000001",
    "timestamp": "2026-09-16T14:30:45.123456",
    "status": "completed",
    "filename": "bridge_crack.jpg",
    "detection_count": 3,
    "average_confidence": 0.87,
    "max_confidence": 0.94,
    "processing_time_ms": 92.1
  }
]
```

**GET /api/v1/inspections/{inspection_id}**

Get specific inspection result.

Response: Same as POST /api/v1/inspect

### Results

**GET /api/v1/results/{inspection_id}**

Get result JSON.

**GET /api/v1/results/{inspection_id}/original**

Get original uploaded image (JPEG).

**GET /api/v1/results/{inspection_id}/annotated**

Get annotated result image with crack overlays (JPEG).

## Error Responses

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Uploaded file is not a supported image."
  }
}
```

Common HTTP status codes:
- 200: Success
- 400: Invalid request/image
- 413: File too large
- 422: Invalid parameter value
- 500: Server error
- 503: Model not loaded

## Testing

### Run unit tests:

```bash
cd backend
pytest tests/
```

### Run smoke test:

```bash
# In a separate terminal, start the backend first
python ../tools/test_backend.py
```

This verifies:
- Health endpoint works
- Model loads correctly
- Real inference runs
- Results are generated
- Images can be retrieved

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app setup
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_health.py    # Health endpoints
│   │   ├── routes_system.py    # System status
│   │   ├── routes_inspection.py # Inspection endpoint
│   │   └── routes_results.py   # Result retrieval
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Pydantic settings
│   │   └── logging_config.py   # Logging setup
│   ├── schemas/
│   │   └── __init__.py         # Pydantic models
│   ├── services/
│   │   ├── __init__.py         # Inference service
│   │   ├── inspection_service.py # Workflow
│   │   └── result_service.py   # Result retrieval
│   └── utils/
│       ├── __init__.py         # Image utilities
│       └── file_utils.py       # File utilities
├── tests/
│   ├── test_health.py
│   ├── test_system.py
│   └── test_inspection.py
├── storage/                    # Generated at runtime
│   ├── uploads/
│   └── inspections/
├── requirements.txt
├── .env.example
├── run_backend.bat
└── run_backend.sh
```

## Model Loading

The model loads ONCE at application startup:

1. Backend starts
2. `app/main.py` runs startup event
3. `InferenceService` loads `best.pt`
4. Model is kept in memory for all requests
5. On shutdown, model is released

No model reloading happens per request.

## Performance Notes

- Model: YOLOv8n-Seg (small, ~6.3M parameters)
- GPU: NVIDIA RTX 4050 (6GB VRAM)
- Typical inference time: 30-50ms on GPU
- Supports half-precision inference if GPU available
- Inference locked to prevent concurrent GPU issues

## CUDA & Device Handling

- Automatically detects CUDA availability
- Falls back to CPU if CUDA unavailable
- Device specified via `AEROINSPECT_DEVICE` (0 = first GPU)
- GPU memory monitored but not forced to full utilization

```bash
# Check CUDA status:
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Next.js Frontend Integration

The frontend can consume the API:

```javascript
// Upload image
const formData = new FormData();
formData.append('file', imageFile);
formData.append('confidence', 0.25);

const response = await fetch('http://localhost:8000/api/v1/inspect', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.inspection_id);
console.log(result.detections);

// Retrieve annotated image
const img = document.createElement('img');
img.src = `http://localhost:8000/api/v1/results/${result.inspection_id}/annotated`;
```

## Logging

Logs are written to `logs/aeroinspect.log`:

```
[2026-09-16 14:30:45] INFO     aeroinspect - Loading model from D:\...\best.pt
[2026-09-16 14:30:48] INFO     aeroinspect - Model loaded successfully on device: cuda:0
[2026-09-16 14:30:52] INFO     aeroinspect - [INSPECTION] INS-20260916-000001 - Starting inspection
[2026-09-16 14:30:54] INFO     aeroinspect - [INSPECTION] INS-20260916-000001 - COMPLETED (2145.3ms total)
```

## Troubleshooting

### Model not loading
- Verify `AEROINSPECT_MODEL_PATH` in `.env` points to `best.pt`
- Check file exists: `dir D:\Projects\AeroInspectAI\experiments\exp001_yolov8n_seg\runs\baseline\weights\best.pt`
- Verify Ultralytics and PyTorch installed: `pip list | grep -E "ultralytics|torch"`

### CUDA not detected
- Verify CUDA installed: `nvidia-smi`
- Verify PyTorch built with CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Backend still works on CPU (slower)

### FileNotFoundError on startup
- Ensure paths in `.env` are absolute paths
- Use Windows backslashes or forward slashes (both work)
- Check file permissions

### Out of Memory (OOM)
- Reduce `AEROINSPECT_IMAGE_SIZE` in requests
- Reduce batch size if implementing concurrent requests
- Check GPU memory: `nvidia-smi`

### CORS errors from frontend
- Verify `AEROINSPECT_CORS_ORIGINS` includes frontend URL
- Default: `["http://localhost:3000"]`

## Future Enhancements

- [ ] Redis caching for inference results
- [ ] Celery task queue for async inference
- [ ] PostgreSQL for persistent inspection history
- [ ] WebSocket live telemetry streaming
- [ ] Multi-model support
- [ ] Batch inference
- [ ] Model versioning
- [ ] A/B testing
- [ ] Inference statistics dashboard
- [ ] Cloud storage integration (S3, GCS)

## API Documentation

Full interactive API documentation available at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## License

Part of AeroInspectAI project.

## Support

For issues or questions:
1. Check logs in `logs/aeroinspect.log`
2. Run smoke test: `python ../tools/test_backend.py`
3. Verify endpoints via Swagger UI: http://localhost:8000/docs
