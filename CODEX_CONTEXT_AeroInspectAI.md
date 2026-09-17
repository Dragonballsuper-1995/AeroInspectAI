# AeroInspectAI — Complete Codex Project Context

> **Purpose:** This document is the authoritative onboarding/context document for Codex working on the AeroInspectAI Drones Project.
>
> **Current milestone:** ML experimentation completed through **EXP002**.
>
> **Important current state:** The **frontend and backend have NOT been built yet**. Do not assume an existing Next.js or FastAPI application exists.

---

## 1. Project Identity

### Project name

**AeroInspectAI**

### Project type

Autonomous drone-based structural inspection system with AI-assisted crack detection/segmentation.

### Primary objective

Build an end-to-end system in which a drone can eventually inspect structures using autonomous flight/guidance, capture images, and use a computer-vision model to detect and segment structural cracks.

The project has two major sides:

1. **AI / Computer Vision**
   - Crack detection and segmentation from inspection imagery.
   - YOLO-based segmentation model.
   - Experimental training and evaluation.
   - Later improvements through error analysis, threshold tuning, model scaling, etc.

2. **Drone / Software System**
   - Inspection workflow.
   - Waypoint following.
   - Orbit following.
   - Guidance model.
   - Backend inference/API.
   - Frontend inspection dashboard.
   - Future telemetry / autonomous-flight integration.

The immediate software objective is to build a stable inspection dashboard and inference backend around the already completed ML work, while keeping the architecture compatible with future drone telemetry and autonomous guidance.

---

# 2. Current Project Status

## Completed

- Dataset preparation and validation.
- Final dataset creation.
- Dataset quality audit.
- Baseline YOLOv8n-Seg training.
- Baseline validation and held-out test evaluation.
- EXP002 hyperparameter experimentation.
- Initial ML analysis/plots.
- Decision to pause additional model improvement temporarily and move toward the Review-1 dashboard/software milestone.

## NOT completed yet

- Backend.
- Frontend.
- Frontend/backend integration.
- Production inference API.
- Inspection history database/storage.
- Dashboard analytics implementation.
- Real-time telemetry integration.
- Drone simulator integration.
- End-to-end drone demonstration.

### Critical instruction

**Do not claim that a frontend or backend already exists.**

Codex must inspect the repository before creating anything and must preserve the ML work that already exists.

---

# 3. Repository

Primary project root:

```text
D:\Projects\AeroInspectAI
```

Codex should treat this as the project root when working on the local Windows environment.

Expected important directories currently include:

```text
AeroInspectAI/
├── datasets/
├── experiments/
├── ...
```

The repository may evolve as the frontend/backend are created.

### Do not reorganize the entire repository blindly

Before creating a new architecture:

1. Inspect the existing repository.
2. Identify existing scripts, notebooks, experiments, documentation and configuration.
3. Preserve useful existing work.
4. Propose structural changes before making destructive changes.
5. Avoid moving large amounts of existing ML data unless explicitly required.

---

# 4. Development Environment

The known development environment is:

- OS: Windows 11
- Shell: PowerShell / Windows terminal
- Python: **3.14.7**
- PyTorch: **2.10.0+cu128**
- Ultralytics: **8.4.152**
- GPU: **NVIDIA GeForce RTX 4050 Laptop GPU**
- VRAM: **6 GB**
- CUDA: available
- Python virtual environment:

```text
D:\Projects\AeroInspectAI\.venv
```

### Environment preservation rule

Do not casually replace:

- Python version
- PyTorch
- CUDA configuration
- Ultralytics
- existing virtual environment
- GPU-related packages

If a dependency change is genuinely necessary, explain why, make the smallest compatible change, and verify that the existing ML environment still works.

---

# 5. Dataset

## Raw dataset

```text
D:\Projects\AeroInspectAI\datasets\crack-seg
```

### IMPORTANT

The raw dataset must remain untouched.

Do not:

- overwrite raw images
- rename raw files
- rewrite raw labels
- silently delete samples
- change the raw train/val/test split
- modify annotations in-place

If preprocessing or filtering is needed, create a derived dataset.

---

# 6. Final Dataset

Final dataset:

```text
D:\Projects\AeroInspectAI\datasets\aeroinspect_crack_v1
```

Dataset YAML:

```text
D:\Projects\AeroInspectAI\datasets\aeroinspect_crack_v1\aeroinspect_crack_v1.yaml
```

Dataset composition:

| Split | Images |
|---|---:|
| Train | 3717 |
| Validation | 199 |
| Test | 112 |
| **Total** | **4028** |

Single class:

```text
crack
```

An earlier quality scan covered 4029 images / 5290 polygons before finalization.

Quality findings:

- 4024 valid without warnings
- 5 valid with warnings
- warnings included 5 very small polygons
- 1 empty label
- 0 duplicate polygons
- 0 hard annotation errors

One confirmed bad sample was excluded from the finalized dataset:

```text
D:\Projects\AeroInspectAI\datasets\crack-seg\images\val\3513.rf.782300f38d0a008b3340e54b643e713e.jpg
```

The raw dataset itself was not modified.

---

# 7. ML Model

Primary model family used:

**YOLOv8n-Seg**

Task:

**Instance segmentation of structural cracks**

Current known baseline checkpoint:

```text
D:\Projects\AeroInspectAI\experiments\exp001_yolov8n_seg\runs\baseline\weights\best.pt
```

Last checkpoint:

```text
D:\Projects\AeroInspectAI\experiments\exp001_yolov8n_seg\runs\baseline\weights\last.pt
```

Baseline run directory:

```text
D:\Projects\AeroInspectAI\experiments\exp001_yolov8n_seg\runs\baseline
```

### Model loading rule

When implementing real inference:

- load a real trained checkpoint;
- do not create fake detections;
- do not hardcode detection results;
- do not fabricate confidence scores;
- do not fabricate crack counts;
- do not fabricate segmentation masks;
- do not substitute random/sample data while claiming it is model inference.

If the correct production checkpoint is ambiguous, inspect the experiment artifacts and documentation first. If it remains ambiguous, ask before silently choosing a model.

---

# 8. EXP001 — Baseline

Model:

```text
YOLOv8n-Seg
```

Training:

- 100 epochs
- approximately 2.57 hours

Known validation metrics:

### Box

- Precision: approximately **0.817**
- Recall: approximately **0.743**
- mAP@0.50: approximately **0.805**
- mAP@0.50:0.95: approximately **0.629**

### Mask

- Precision: approximately **0.744**
- Recall: approximately **0.683**
- mAP@0.50: approximately **0.696**
- mAP@0.50:0.95: approximately **0.236**

### Held-out test

Box:

- Precision: approximately **0.856**
- Recall: approximately **0.676**
- mAP@0.50: approximately **0.753**
- mAP@0.50:0.95: approximately **0.553**

Mask:

- Precision: approximately **0.800**
- Recall: approximately **0.588**
- mAP@0.50: approximately **0.645**
- mAP@0.50:0.95: approximately **0.234**

These values are historical experiment results. Do not overwrite or reinterpret them.

---

# 9. EXP002 — Hyperparameter Experiment

**EXP002 is the latest completed experiment.**

Known configuration:

| Parameter | EXP002 |
|---|---|
| Model | YOLOv8n-Seg |
| Image size | 640 px |
| Batch size | 24 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0005 |
| Augmentation | Medium |
| Epochs | 25 |

Known validation result:

- Precision: **0.7766**
- Recall: **0.6707**
- F1: **0.7198**
- mAP@0.50: **0.6547**
- mAP@0.50:0.95: **0.2161**
- Best epoch: **23**

The test set was **not used for EXP002**.

### Important interpretation rule

EXP002 was a short hyperparameter experiment.

Its results must **not automatically be treated as a replacement for the 100-epoch EXP001 baseline** simply because EXP002 was the latest experiment.

The project currently has:

- a long-trained baseline from EXP001;
- a short EXP002 tuning experiment;
- no separately declared final production model unless the repository explicitly establishes one.

Codex must inspect the actual experiment artifacts before deciding which checkpoint should power the dashboard.

---

# 10. Current ML Decision

The project is temporarily moving forward to software/dashboard development.

Additional model work is intentionally deferred until after the current Review-1 milestone.

Do not automatically:

- start another training run;
- switch to YOLOv8s;
- perform threshold tuning;
- modify the dataset;
- perform extensive error analysis;
- evaluate additional models.

Those are future ML tasks unless explicitly requested.

### Current priority

**Build a stable, demonstrable inspection software stack around the completed ML work.**

Review-1/demo stability is more important than unnecessary architectural overengineering.

---

# 11. Important ML Metrics Context

The project has observed that segmentation is substantially harder than box detection.

Known baseline pattern:

- box mAP is considerably stronger;
- mask mAP@0.50:0.95 is much lower;
- recall is a notable area for future improvement.

Plots have included:

- training loss curves;
- precision/recall;
- F1-confidence curves;
- precision-confidence curves;
- PR curves;
- confusion matrices;
- normalized confusion matrices.

Codex must preserve experiment outputs and should not overwrite plots or metrics.

When displaying metrics in the dashboard, clearly distinguish:

- box metrics;
- mask metrics;
- validation metrics;
- test metrics;
- experiment metrics.

Do not mix metrics from different experiments.

---

# 12. Future System Architecture

The intended long-term system is conceptually:

```text
                 ┌─────────────────────────┐
                 │      Drone / Simulator   │
                 │                         │
                 │ Waypoints / Orbit /     │
                 │ Guidance / Telemetry    │
                 └────────────┬────────────┘
                              │
                              │ Images + Telemetry
                              ▼
                 ┌─────────────────────────┐
                 │       FastAPI Backend   │
                 │                         │
                 │ Inspection API          │
                 │ YOLO Inference          │
                 │ Results                 │
                 │ History                 │
                 │ Telemetry / WebSocket   │
                 └────────────┬────────────┘
                              │
                    REST / WebSocket
                              │
                              ▼
                 ┌─────────────────────────┐
                 │       Next.js Frontend  │
                 │                         │
                 │ Dashboard               │
                 │ New Inspection          │
                 │ Results                 │
                 │ Inspection History      │
                 │ Analytics               │
                 │ Experiments             │
                 │ System                  │
                 └─────────────────────────┘
```

This is the target architecture, not an indication that these components already exist.

---

# 13. Frontend — Current Status

### Status

**NOT BUILT YET.**

No existing frontend should be assumed.

The planned frontend is a modern Next.js dashboard for inspection operations.

Potential pages/modules:

1. Overview
2. New Inspection
3. Inspection Results
4. Inspection History
5. Analytics
6. Experiments
7. System Status
8. Settings

These are planned modules and may be adjusted after repository inspection.

### Frontend goals

The UI should be:

- clean
- modern
- professional
- responsive
- suitable for an engineering/research project
- visually useful for inspection imagery
- easy to demonstrate during Review-1
- structured so that future real-time telemetry can be integrated.

Do not overbuild the frontend before the core inspection flow works.

---

# 14. Backend — Current Status

### Status

**NOT BUILT YET.**

The intended backend is FastAPI.

Initial architecture should separate:

```text
API layer
    ↓
Inspection / business logic
    ↓
Inference service
    ↓
YOLO model
```

The model should be loaded once and reused rather than loaded for every request.

---

# 15. Planned Backend API

Initial REST API concept:

```text
GET  /api/v1/health
GET  /api/v1/system/status

POST /api/v1/inspect

GET  /api/v1/inspections
GET  /api/v1/inspections/{inspection_id}

GET  /api/v1/inspections/{inspection_id}/image
GET  /api/v1/inspections/{inspection_id}/result
```

Future real-time endpoint:

```text
WS /api/v1/ws/telemetry
```

The exact API contract should be finalized by inspecting the repository and implementing the smallest useful MVP.

---

# 16. Inspection Flow

The intended initial inspection flow is:

```text
User uploads inspection image
            ↓
Frontend sends image to FastAPI
            ↓
Backend validates upload
            ↓
InferenceService runs YOLOv8-Seg
            ↓
Model returns detections + segmentation
            ↓
Backend calculates structured result
            ↓
Annotated image generated
            ↓
Inspection result stored
            ↓
JSON response returned
            ↓
Frontend displays:
  - original image
  - segmentation overlay
  - crack count
  - confidence
  - bounding boxes / masks
  - relevant measurements
```

The first implementation should focus on making this flow reliable.

---

# 17. Inference Result Requirements

A real inspection result should contain structured information such as:

```json
{
  "inspection_id": "...",
  "image": "...",
  "detections": [
    {
      "class_id": 0,
      "class_name": "crack",
      "confidence": 0.91,
      "bbox": {
        "x1": 100,
        "y1": 120,
        "x2": 300,
        "y2": 260
      },
      "polygon": [],
      "area": 0
    }
  ],
  "summary": {
    "crack_count": 1,
    "mean_confidence": 0.91
  }
}
```

This is a conceptual contract, not a requirement to use these exact field names.

Codex should design the actual schema cleanly and document it.

### Important

Only expose measurements that can actually be calculated from the model/image.

Do not present unsupported physical measurements such as real crack width in millimetres unless camera calibration/depth/reference information exists.

Pixel measurements and model-derived quantities are acceptable when explicitly labeled.

---

# 18. Severity

The dashboard may eventually show a severity classification.

However, **severity must not be presented as a scientifically validated structural-safety assessment unless an actual validated severity model/rule exists.**

If an initial UI needs severity for demonstration, it should be clearly labeled as a project-defined heuristic and documented.

Never imply:

> "This crack is structurally unsafe."

based solely on YOLO segmentation.

---

# 19. Backend Requirements

The initial FastAPI implementation should include:

### Core

- FastAPI
- Pydantic models
- clean application structure
- configuration through environment variables where appropriate
- CORS for local frontend development
- structured error responses
- logging
- health endpoint
- system status endpoint

### Inference

- dedicated `InferenceService`
- model loaded once at application startup
- CUDA when available
- CPU fallback
- configurable confidence threshold
- configurable image size
- actual YOLO inference
- segmentation polygons/masks
- bounding boxes
- confidence scores
- annotated output image

### Storage

Start simple.

A filesystem-based result store is acceptable for the MVP.

Do not introduce PostgreSQL/Redis/object storage unless the project actually needs it.

Each inspection should have a stable ID and associated result/output files.

### Security basics

- validate uploaded file type;
- limit upload size;
- use safe filenames;
- avoid path traversal;
- do not execute uploaded content;
- reject unsupported formats;
- return useful errors.

---

# 20. Frontend/Backend Development Principle

Build the system in vertical slices.

Preferred order:

### Slice 1

```text
FastAPI health endpoint
        ↓
Next.js can reach backend
```

### Slice 2

```text
Image upload
        ↓
Real YOLO inference
        ↓
JSON result
```

### Slice 3

```text
Annotated image
        ↓
Results page
```

### Slice 4

```text
Inspection persistence
        ↓
History page
```

### Slice 5

```text
Analytics
        ↓
Experiment visualization
```

### Slice 6

```text
Telemetry/WebSocket foundation
```

Do not build eight disconnected pages with fake data before the core inspection flow works.

---

# 21. Demo Strategy

The first software milestone should support a convincing demo:

```text
Open dashboard
      ↓
Select inspection image
      ↓
Run inspection
      ↓
Show processing state
      ↓
Display original image
      ↓
Display segmentation overlay
      ↓
Show crack detections
      ↓
Show confidence/count statistics
      ↓
Save inspection
      ↓
Open inspection history
```

This should work using the real trained model.

A polished demo is more valuable at this stage than excessive backend complexity.

---

# 22. Planned Dashboard Sections

## Overview

Show:

- system status
- model status
- number of inspections
- recent inspections
- basic crack statistics
- current ML model information

## New Inspection

Allow:

- image upload
- preview
- run inspection
- processing indicator
- error state

## Results

Show:

- original image
- segmentation result
- overlay
- crack count
- confidence
- individual detections
- bounding boxes
- segmentation polygons/masks
- pixel-based area where available

## Inspection History

Show:

- inspection ID
- date/time
- image
- crack count
- confidence
- status

## Analytics

Potentially show:

- inspection counts
- crack-count distribution
- confidence distribution
- segmentation statistics
- model metrics

Do not invent historical data.

## Experiments

Display actual experiment metadata/results, for example:

- EXP001
- EXP002
- model
- image size
- optimizer
- learning rate
- batch size
- epochs
- validation metrics
- test metrics where available

Do not mix validation and test values.

---

# 23. Experiments Page Rules

Experiment data should be traceable to actual artifacts.

At minimum, distinguish:

```text
Experiment ID
Model
Configuration
Dataset
Training duration
Epochs
Validation metrics
Test metrics
Checkpoint path
Notes
```

If a value cannot be verified from the repository, do not fabricate it.

If Codex extracts experiment metadata from files, preserve the source filename/path in documentation or internal metadata when practical.

---

# 24. Drone / Guidance Roadmap

The broader project includes autonomous drone guidance work.

Future capabilities include:

- predefined 3-D waypoints
- waypoint follower
- orbit follower
- guidance model
- inspection trajectory
- telemetry
- image capture along trajectory
- AI inspection of captured imagery

The architecture should therefore leave room for:

```text
Position
Velocity
Attitude
Battery
Flight mode
Waypoint state
Inspection state
Camera/image events
```

But these do **not** need to be fully implemented in the first dashboard/backend milestone.

---

# 25. Simulink / Autonomous Drone Context

The project includes MATLAB/Simulink autonomous-drone laboratory work.

Relevant lab direction includes:

- waypoint follower
- orbit follower
- guidance model

The eventual software architecture should be compatible with receiving simulated or real telemetry from a drone/guidance system.

Do not unnecessarily couple the first FastAPI implementation to MATLAB/Simulink.

Use a clean interface so telemetry can be integrated later.

---

# 26. Git / File Safety

Codex must be conservative with destructive operations.

### Never delete without explicit approval

- datasets
- experiment folders
- trained weights
- evaluation results
- plots
- notebooks
- audit outputs
- existing source code
- configuration files

### Never overwrite trained weights accidentally

If a new experiment is run, use a new experiment/run directory.

Never silently replace:

```text
best.pt
last.pt
```

from a previous experiment.

---

# 27. Dataset Safety Rules

These are hard constraints.

Codex must NOT:

- modify the raw dataset;
- silently alter annotations;
- silently alter train/val/test membership;
- regenerate the finalized dataset without explicit approval;
- delete the excluded-sample record;
- move dataset files merely to make the application architecture look cleaner;
- use the test set for tuning;
- report test performance for EXP002 when EXP002 did not use the test set.

Any new preprocessing must produce a separate derived artifact.

---

# 28. ML Integrity Rules

Codex must NOT:

- fabricate metrics;
- fabricate detections;
- fabricate segmentation masks;
- fabricate model confidence;
- claim a model was trained when it was not;
- claim test evaluation when it did not occur;
- silently change thresholds and call the resulting metrics official;
- silently select a new "best model";
- start long training jobs without explicit user approval;
- change model architecture without explicit user approval;
- present heuristic severity as structural engineering truth.

When the application is running in demo mode, demo/mock data must be explicitly labeled as demo/mock.

---

# 29. Test Set Protection

The test set is held out for final evaluation.

Therefore:

**Do not use the test set to tune the model.**

In particular, do not use test performance to select:

- confidence threshold;
- augmentation;
- model size;
- optimizer;
- learning rate;
- architecture;
- preprocessing.

The test set should remain an evaluation resource.

---

# 30. Coding Style

Prefer:

- clear names;
- small modules;
- type hints;
- Pydantic schemas for API data;
- service separation;
- reusable utilities;
- readable functions;
- useful error messages;
- meaningful logging;
- configuration over hardcoding.

Avoid:

- giant files;
- duplicated logic;
- unnecessary abstractions;
- premature microservices;
- unnecessary databases;
- magic constants;
- fake API responses presented as real results.

---

# 31. Frontend Engineering Rules

When building the frontend:

- use a coherent component architecture;
- keep API communication separate from presentation;
- centralize API types/contracts;
- handle loading states;
- handle empty states;
- handle errors;
- handle backend-unavailable states;
- make image visualization a first-class feature;
- ensure responsive layout;
- avoid excessive animation;
- avoid UI complexity that does not support inspection workflow.

The frontend must remain usable even when the backend is temporarily unavailable.

---

# 32. Backend Engineering Rules

When building the backend:

- keep inference isolated from routing;
- avoid loading the model per request;
- avoid global mutable state where unnecessary;
- validate inputs;
- return consistent response structures;
- log inference failures;
- expose health/system status;
- make device selection observable;
- document API endpoints;
- keep the inference layer replaceable so future models can be substituted.

---

# 33. Configuration

Do not hardcode machine-specific paths throughout the source code.

For example, avoid scattering:

```text
D:\Projects\AeroInspectAI\...
```

through application modules.

Prefer configuration such as:

```text
MODEL_PATH
DATASET_PATH
RESULTS_PATH
CONFIDENCE_THRESHOLD
IMAGE_SIZE
DEVICE
```

However, retain repository-relative defaults where they make development easier.

---

# 34. Local Development Target

Expected local architecture:

```text
Frontend:
http://localhost:3000

Backend:
http://localhost:8000
```

The exact ports may be configurable.

CORS should permit the local frontend to communicate with the backend.

---

# 35. Suggested Backend Structure

A reasonable initial structure is:

```text
backend/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── api/
│   │   └── routes/
│   │       ├── health.py
│   │       ├── inspections.py
│   │       └── system.py
│   ├── schemas/
│   │   ├── inspection.py
│   │   └── system.py
│   ├── services/
│   │   ├── inference.py
│   │   └── inspection.py
│   └── storage/
│       └── ...
├── tests/
├── requirements.txt
└── README.md
```

This is a recommendation, not an instruction to create all files blindly.

Inspect first.

---

# 36. Suggested Frontend Structure

A reasonable initial structure is:

```text
frontend/
├── app/
│   ├── page.tsx
│   ├── inspections/
│   ├── results/
│   ├── analytics/
│   ├── experiments/
│   └── system/
├── components/
├── lib/
│   ├── api/
│   └── types/
├── public/
├── package.json
└── README.md
```

Again, inspect the repository and current Next.js version before implementation.

---

# 37. API Contract Principle

The frontend should never depend on undocumented backend internals.

Document:

- endpoint
- HTTP method
- request
- response
- errors
- status codes
- image/result URLs

The API contract should be stable enough that frontend and backend can be developed independently.

---

# 38. Mock Data Rule

Mock data is acceptable **only** for UI development when the real backend endpoint is not yet implemented.

When mock data is used:

- isolate it behind an adapter;
- make it obvious in code;
- label demo/mock mode;
- do not mix it with real inspection results;
- make replacing it with the real API straightforward.

Once the real inference API exists, the main inspection flow should use real inference.

---

# 39. Testing

At minimum, backend development should eventually include:

### Unit tests

- configuration
- result schema
- validation
- result transformation

### API tests

- health endpoint
- system status
- invalid upload
- valid upload
- inspection response

### Inference smoke test

Run one real image through the model and verify:

- inference succeeds;
- at least the response schema is correct;
- model/device is reported correctly;
- annotated output is generated;
- no fake result path is being used.

### Frontend

Verify:

- application starts;
- pages render;
- API loading state;
- API error state;
- successful inspection flow;
- result visualization.

---

# 40. Agent Workflow

Codex should follow this workflow for every significant task:

## Step 1 — Inspect

Read:

- repository structure;
- relevant source files;
- package files;
- Python environment;
- experiment artifacts;
- existing documentation.

## Step 2 — Plan

Explain:

- what will change;
- which files will be created/modified;
- dependencies;
- risks;
- verification method.

## Step 3 — Implement

Make the smallest coherent change.

## Step 4 — Test

Run relevant:

- lint;
- type checks;
- unit tests;
- API tests;
- smoke tests;
- build commands.

## Step 5 — Verify

Confirm that existing ML assets were not damaged.

## Step 6 — Report

Summarize:

- files changed;
- functionality added;
- tests run;
- test results;
- known limitations;
- next recommended step.

---

# 41. Do Not Ask Unnecessary Questions

Codex should make reasonable implementation decisions when they are low-risk and reversible.

Ask the user when a decision would:

- delete or modify existing data;
- change the dataset;
- change the ML model;
- start expensive training;
- change the intended architecture substantially;
- expose sensitive information;
- require a paid external service;
- choose between ambiguous trained checkpoints where the choice affects official results.

---

# 42. No Paid Infrastructure by Default

The project is intended to run locally during development.

Do not introduce paid services by default.

Avoid requiring:

- DigitalOcean
- paid cloud GPU
- paid databases
- paid object storage
- paid API providers

unless explicitly requested.

---

# 43. External Dependencies

Prefer mature, well-supported libraries.

Do not add a dependency simply because it makes one small feature easier.

Before adding a major dependency:

1. determine whether the existing stack already provides the functionality;
2. consider maintenance cost;
3. explain why it is needed;
4. update documentation.

---

# 44. Documentation Requirements

Every major component should have documentation.

At minimum:

```text
README.md
backend/README.md
frontend/README.md
```

Eventually document:

- project setup;
- environment;
- model location;
- starting backend;
- starting frontend;
- API endpoints;
- inference behavior;
- experiment results;
- architecture;
- troubleshooting.

---

# 45. Important Distinctions

Codex must maintain these distinctions:

### Training vs inference

Training scripts/experiments are not the same as the production inference service.

### Validation vs test

Validation can be used during development/tuning.

Test is held out for final evaluation.

### Box vs mask

Detection bounding-box metrics and segmentation mask metrics are different.

### Pixel measurement vs physical measurement

Pixel area/length is not automatically millimetres/centimetres.

### Heuristic severity vs engineering assessment

A UI severity heuristic is not a structural safety diagnosis.

### Demo/mock vs real inference

Mock results must never be presented as actual model output.

---

# 46. Current Priority Roadmap

## Phase 1 — Repository + architecture

- inspect repository
- establish project instructions
- document ML artifacts
- create clean backend/frontend boundaries

## Phase 2 — Backend MVP

- FastAPI
- health
- system status
- model loading
- inference service
- image inspection endpoint

## Phase 3 — Frontend MVP

- Next.js
- dashboard shell
- image upload
- inspection workflow
- result visualization

## Phase 4 — Persistence

- inspection IDs
- saved results
- history

## Phase 5 — Analytics

- actual inspection analytics
- experiment metrics
- ML result visualization

## Phase 6 — Telemetry

- WebSocket foundation
- simulated telemetry
- inspection state

## Phase 7 — Drone integration

- waypoint state
- orbit state
- guidance integration
- image capture events

## Phase 8 — End-to-end demo

```text
Mission
 ↓
Drone/simulator
 ↓
Image capture
 ↓
Backend inference
 ↓
Segmentation
 ↓
Dashboard
 ↓
Inspection record
```

---

# 47. Future ML Roadmap

After the software/review milestone, possible ML work includes:

- error analysis;
- confidence-threshold analysis;
- longer EXP002/selected configuration training;
- YOLOv8s comparison;
- augmentation experiments;
- class/annotation analysis;
- segmentation-quality improvements;
- final held-out test evaluation;
- model optimization;
- deployment/edge inference.

These should be treated as future experiments, not automatically executed.

---

# 48. Current Known ML Target

A project target previously discussed is approximately:

```text
F1 >= 0.88
```

This is a project goal, **not the current achieved performance**.

The current known EXP002 validation F1 is:

```text
0.7198
```

Do not represent the target as an achieved result.

---

# 49. What Codex Should Do First

When first connected to the repository, Codex should NOT immediately build the frontend and backend.

First perform a repository audit.

The first task should be conceptually:

> "Inspect the AeroInspectAI repository and understand the current state. We have completed ML experimentation through EXP002. The frontend and backend do not exist yet. Identify the existing dataset, experiment, model, documentation, scripts, environment configuration, and any existing application code. Do not modify files. Report the current structure, important artifacts, risks, and a proposed implementation plan for building the FastAPI backend and Next.js frontend."

After the audit, implementation can proceed in small milestones.

---

# 50. First Implementation Milestone

After the audit, the first actual implementation should be:

### Backend foundation

- create FastAPI application;
- configuration;
- health endpoint;
- system status;
- CORS;
- logging;
- model discovery/loading;
- inference service skeleton;
- tests.

Then implement:

```text
POST /api/v1/inspect
```

using the real YOLO segmentation model.

Only after this works should the frontend inspection workflow be connected.

---

# 51. Definition of Done — Initial MVP

The initial MVP is successful when:

1. Backend starts locally.
2. Backend reports model/device status.
3. Frontend starts locally.
4. Frontend can select an image.
5. Image reaches backend.
6. Backend performs real YOLOv8-Seg inference.
7. Backend returns structured detections.
8. Segmentation output is available.
9. Annotated image is generated.
10. Frontend displays the result.
11. Inspection can be saved.
12. Inspection history can be viewed.
13. Errors are handled cleanly.
14. No ML artifacts were damaged.
15. No fake data is used in the real inspection path.

---

# 52. Hard "DO NOT" List

Codex must not:

- assume frontend exists;
- assume backend exists;
- rewrite the project from scratch without inspection;
- delete existing ML experiments;
- modify raw dataset;
- silently change dataset splits;
- use the test set for tuning;
- overwrite trained weights;
- fabricate ML metrics;
- fabricate inference results;
- claim EXP002 test performance;
- start expensive training without explicit approval;
- change Python/CUDA unnecessarily;
- introduce paid infrastructure without approval;
- expose secrets;
- hardcode credentials;
- hardcode machine-specific paths throughout application code;
- present mock data as real inference;
- claim crack severity is a structural safety diagnosis;
- overengineer the first MVP.

---

# 53. Codex Working Principle

The project should be developed with the following priority:

```text
Correctness
    ↓
Preservation of existing ML work
    ↓
Reliable real inference
    ↓
Simple architecture
    ↓
Good user experience
    ↓
Extensibility
    ↓
Optimization
```

Do not optimize architecture at the expense of a working demo.

---

# 54. Final Context Summary

At the moment Codex takes over:

```text
AeroInspectAI
│
├── Dataset
│   ├── raw dataset preserved
│   └── finalized dataset: 4028 images
│
├── ML
│   ├── EXP001 baseline completed
│   └── EXP002 completed
│
├── Model
│   └── YOLOv8n-Seg
│
├── Frontend
│   └── NOT BUILT
│
├── Backend
│   └── NOT BUILT
│
├── Drone/GCS/Telemetry
│   └── Future integration
│
└── Immediate goal
    └── Build stable backend + frontend inspection MVP
```

### Most important instruction

**Treat the ML experimentation work through EXP002 as the foundation. Do not redo it. Do not damage it. Build the software system around it.**

Start by inspecting the repository, then build the backend and frontend incrementally with real model inference.

