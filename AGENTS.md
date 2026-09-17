# Repository Guidelines

## Project Structure & Module Organization

- `frontend/` is the Next.js 15 dashboard. Routes live in `app/`, shared UI in `components/`, and API types/services in `lib/`.
- `backend/app/` contains FastAPI routes, configuration, schemas, services, and utilities. Tests are in `backend/tests/`.
- `tools/` holds dataset audit, training, calibration, and smoke-test scripts. `configs/` stores ML configuration.
- `datasets/`, `experiments/`, `runs/`, and `weights/` contain large or generated ML assets. Preserve existing splits, checkpoints, metrics, and plots; never overwrite a prior experiment run.
- `backend/storage/`, `backend/logs/`, `.next/`, and `node_modules/` are generated content.

## Build, Test, and Development Commands

Run frontend commands from `frontend/`:

- `npm install` installs pinned dependencies.
- `npm run dev` starts the dashboard at `http://localhost:3000`.
- `npm run build` creates a production build and checks TypeScript.
- `npm run lint` runs the configured Next.js lint task.

Run backend commands from `backend/` after activating the repository `.venv`:

- `pip install -r requirements.txt` installs API and inference dependencies. Preserve the existing CUDA/PyTorch stack.
- `python -m uvicorn app.main:app --reload --port 8000` starts the API and docs.
- `pytest tests/` runs the FastAPI test suite.
- `python ../tools/test_backend.py` smoke-tests a running backend and real inference flow.

## Coding Style & Naming Conventions

Use four spaces, type hints, `snake_case` modules/functions, and `PascalCase` classes in Python. Keep routes thin and workflow behavior in services. For TypeScript, retain strict mode, use two spaces, `PascalCase` components, and `camelCase` variables/functions. Centralize API contracts in `frontend/lib/`; avoid machine-specific paths.

## Testing Guidelines

Pytest uses `test_*.py` files and `test_*` functions. Cover endpoint validation and failures with backend changes. Frontend changes rely on lint/build checks; manually verify loading, empty, error, and backend-unavailable states. Never use the held-out test dataset for tuning.

## Commit & Pull Request Guidelines

This checkout contains no Git history, so no repository-specific convention can be verified. Use concise Conventional Commit subjects such as `feat(backend): add inspection history endpoint`. Keep commits focused. Pull requests should explain behavior and verification, link relevant issues, call out model/dataset/config changes, and include screenshots for UI work or sample API responses for contract changes.

## Security & ML Integrity

Keep secrets and local paths in `.env`; validate uploads and filenames. Do not fabricate detections or metrics, modify raw datasets, relabel heuristic severity as structural assessment, or start long training jobs without explicit approval.
