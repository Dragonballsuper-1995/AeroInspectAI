# AeroInspect AI

AeroInspect AI is a local-first infrastructure inspection console. It combines a Next.js dashboard, a FastAPI inference service, a YOLOv8n-Seg crack model, and an optional Gazebo + ArduPilot SITL workflow for simulated drone captures.

## Repository map

| Area | Purpose |
| --- | --- |
| `frontend/` | Next.js 15 dashboard and inspection workflow |
| `backend/` | FastAPI API, inference service, storage, and tests |
| `experiments/` | Training runs and checkpoint metadata |
| `datasets/` | Dataset files; intentionally ignored by Git |
| `simulation/` | Gazebo world, vehicle model, camera, and mission controller |
| `scripts/` | Windows launcher and process helpers |
| `frontend/public/architecture/` | Saved product architecture artwork |

## Local development

Prerequisites: Node.js, Python, a backend virtual environment, and (optionally) WSL 2 with Gazebo and ArduPilot SITL.

```powershell
# Start the dashboard and FastAPI backend
.\START_AEROINSPECT.bat

# Open the dashboard
Start-Process http://127.0.0.1:3000
```

Frontend-only commands:

```powershell
cd frontend
npm install
npm run dev
npm run lint
npm run build
```

Backend commands:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Copy `backend/.env.example` to `backend/.env` and `frontend/.env.example` to `frontend/.env.local`. Local `.env` files are ignored and must not be committed.

## Model weights

The backend uses the trained checkpoint:

`experiments/exp001_yolov8n_seg/runs/baseline/weights/best.pt`

This is the `EXP001` YOLOv8n-Seg baseline checkpoint (about 6.5 MiB). `backend/app/core/config.py` supplies this path by default, while `AEROINSPECT_MODEL_PATH` can override it. All `.pt` files are ignored, so deployment must provide the checkpoint separately through the target platform's artifact storage or image build.

The root `yolov8n-seg.pt` file is a base Ultralytics weight and is not the checkpoint used by the inspection API.

## Deployment preparation

### Vercel (frontend)

The `frontend/` app is the Vercel-ready part of the project:

1. Create a Vercel project with the repository root set to `frontend/`.
2. Use `npm run build` as the build command; Next.js supplies the output configuration.
3. Set `NEXT_PUBLIC_API_URL` to the public HTTPS URL of the FastAPI service.
4. Configure the backend CORS allow-list with the Vercel deployment origin.

The browser client must not point at `localhost` in production. Local file preview uses browser object URLs and does not require a storage service.

### Hugging Face (backend)

The FastAPI service can be hosted in a Hugging Face Docker Space or another Python container. Before deploying, provide `best.pt`, set `AEROINSPECT_MODEL_PATH` to its container path, expose the API port, and configure `AEROINSPECT_CORS_ORIGINS` for the Vercel origin. The current requirements include PyTorch, Ultralytics, OpenCV, and CUDA-oriented defaults; a CPU Space will run more slowly and may need a smaller runtime image.

The current inspection storage is filesystem-backed. A production deployment should use persistent volume/object storage for uploads, annotated results, and history instead of ephemeral container disk.

### Where Gazebo runs

Gazebo does **not** run inside Vercel serverless functions or a standard Hugging Face inference Space. It needs a long-lived Linux/WSL host with Gazebo rendering or headless rendering, ArduPilot SITL, MAVLink networking, and (for the visible GUI) a display server. Keep Gazebo on a local workstation, dedicated Linux VM, or separate simulation host. Connect it to the deployed backend through an authenticated service/controller boundary; the dashboard can consume its telemetry and captured-frame API, but Vercel and Hugging Face do not replace the simulator host.

## Git and generated files

The root `.gitignore` excludes virtual environments, dependency trees, Next.js output, runtime logs, local storage, datasets, training runs, checkpoints, telemetry logs, and deployment state. Keep example configuration files (`*.env.example`) and source code tracked. Never commit secrets, `.env` files, model weights, or generated inspection uploads.

## Verification

```powershell
cd frontend
npm run lint
npm run build

cd ..\backend
..\.venv\Scripts\python.exe -m pytest tests
```

The Gazebo workflow remains an optional local/SITL integration and is not required for ordinary image inspection through the deployed API.
