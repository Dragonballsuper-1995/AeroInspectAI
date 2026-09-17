# AeroInspectAI — Project Review Guide

**Review scope:** software implementation through the simulation milestone  
**Prepared from:** repository source, experiment records, test results, simulation state, and the five attached laboratory reference images  
**Repository:** `D:\Projects\AeroInspectAI`  
**Status checked:** 17 September 2026

> This document is a review aid. It explains the implemented system in simple terms, but it does not claim that an AI detection is a structural-safety certification. A qualified engineer must make the final safety decision.

## 1. How to read this document

The attached images are laboratory reference material. They describe available drones, equipment, software, and a list of possible experiments. They are not instructions that override the project request, and they are not evidence that every listed experiment has already been completed.

This guide uses the following meanings:

- **Implemented:** present in the repository and supported by code or a saved artifact.
- **Verified:** exercised by an automated test, build, or recorded simulation result.
- **Partially implemented:** the foundation exists, but an important production capability is still missing.
- **Planned:** a reasonable next step, but not currently implemented.
- **Lab reference:** information read from the attached posters; it is hardware/context information, not project completion evidence.

## 2. One-minute project summary

AeroInspectAI is a software system for drone-assisted visual inspection of structures. Its current AI task is to identify and segment visible cracks in an image.

The project combines four layers:

1. **Computer vision model:** a trained Ultralytics YOLOv8n segmentation model detects the class `crack` and produces a confidence score, bounding box, and pixel mask.
2. **FastAPI backend:** receives images, validates them, runs inference, creates annotated images, stores results, and exposes APIs.
3. **Next.js dashboard:** lets a reviewer upload an image, inspect the result, view history and experiment context, and monitor the simulator.
4. **Drone simulation:** Gazebo and ArduPilot SITL provide a simulated vehicle, simulated flight, telemetry, camera frames, and an autonomous inspection mission.

The current end-to-end simulation flow is:

```text
Gazebo camera frame
        ↓
WSL mission controller reads the frame and MAVLink telemetry
        ↓
FastAPI receives the frame
        ↓
YOLOv8n-Seg performs crack inference
        ↓
FastAPI stores JSON, original image, and annotated image
        ↓
Next.js displays telemetry, latest frame, and linked inspection result
```

The project has reached a meaningful simulation milestone: the saved simulation state records a completed mission with five captures, confirmed landing, and linked inspection results. This is not yet a real-drone deployment.

## 3. What problem the project is solving

Manual inspection of bridges, walls, concrete panels, and similar structures can be slow, repetitive, difficult to access, and dependent on the quality of the inspector's photographs. AeroInspectAI is intended to assist this process by:

- collecting images from a drone or an uploaded image source;
- finding visible crack-like regions automatically;
- showing the result as an annotated image instead of only raw numbers;
- retaining the result for later review;
- associating a simulated capture with the vehicle's position and mission waypoint; and
- eventually supporting repeatable autonomous inspection missions.

The current system is a **visual inspection assistant**. It does not estimate structural load capacity, crack depth, material strength, remaining service life, or legal compliance.

## 4. Overall architecture

```text
                         Review operator
                              │
                              ▼
                     Next.js web dashboard
                              │ HTTP + WebSocket
                              ▼
                        FastAPI backend
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
       Inspection API   Simulation API     Result storage
             │                │                │
             ▼                ▼                │
       YOLOv8n-Seg     WSL mission controller │
                              │                │
                         MAVLink/TCP          │
                              ▼                │
                       ArduPilot SITL         │
                              │                │
                              ▼                │
                    Gazebo Harmonic world ────┘
```

### Responsibility boundaries

| Component | Responsibility | What it should not do |
|---|---|---|
| YOLOv8n-Seg | Visual crack detection and segmentation | Fly the vehicle or certify safety |
| FastAPI | Application API, inference workflow, storage, simulator bridge | Directly control low-level motor stabilization |
| Mission controller | MAVLink connection, mission state, waypoints, camera capture | Own the web UI or duplicate model logic |
| ArduPilot | Arming, attitude, altitude, navigation, RTL, landing | Run the web dashboard |
| Gazebo | Simulated world, physics, vehicle and camera sensor | Become the application database |
| Next.js | Human-facing review and mission monitoring | Invent telemetry when the simulator is unavailable |

## 5. Implemented machine-learning work

### 5.1 Dataset preparation

The repository contains a quality-checked crack segmentation dataset named `aeroinspect_crack_v1`.

Recorded split counts are:

| Split | Images |
|---|---:|
| Training | 3,717 |
| Validation | 199 |
| Held-out test | 112 |
| Total final dataset | 4,028 |
| Excluded as bad | 1 |
| Warning samples recorded | 5 |
| Cross-split duplicate groups | 0 |

The held-out test split is kept separate from the training and tuning workflow. This is important when explaining the credibility of evaluation numbers during a review.

Relevant artifacts include:

- `datasets/aeroinspect_crack_v1/aeroinspect_crack_v1.yaml`
- `audit_results/final_dataset/reports/final_dataset_summary.json`
- `audit_results/final_dataset/reports/manifest.json`
- `audit_results/final_dataset/reports/excluded_samples.json`
- `tools/dataset_quality_scan.py`
- `tools/finalize_aeroinspect_dataset.py`

### 5.2 Model and training

The implemented model is Ultralytics **YOLOv8n-Seg**, a small segmentation model. The single target class is `crack`.

The main trained checkpoint used by the backend is:

```text
experiments/exp001_yolov8n_seg/runs/baseline/weights/best.pt
```

The experiment record identifies the following training context:

- 100-epoch baseline experiment;
- 512-pixel training image size;
- batch size 4;
- CUDA training on an NVIDIA GeForce RTX 4050 Laptop GPU;
- roughly 6 GB GPU memory;
- a saved `best.pt` checkpoint and epoch checkpoints;
- a completed held-out test evaluation record.

The dashboard also contains a review summary for a later validation-only hyperparameter screen, EXP002. That summary must not be described as a held-out test result. The repository intentionally displays this distinction in the Experiments page.

### 5.3 What the model returns

For each detected crack, the inference service extracts:

- class ID and class name;
- confidence score;
- bounding box (`x1`, `y1`, `x2`, `y2`);
- segmentation polygon, when available;
- mask area in pixels; and
- mask area ratio relative to the image.

The service also records image size, threshold values, device, model name, and inference time. The result is therefore more useful than a simple “crack/no crack” answer.

## 6. Implemented backend

### 6.1 Startup and model loading

`backend/app/main.py` creates the FastAPI application and loads the inference service during application startup. The application reports an error if the model cannot be initialized.

`backend/app/core/config.py` centralizes configuration such as:

- checkpoint path;
- confidence threshold: `0.25` by default;
- IoU threshold: `0.70` by default;
- inference image size: `640` by default;
- maximum upload size: `20 MB`;
- storage directories; and
- allowed frontend origins.

The model device is selected from CUDA availability and configuration. If CUDA is unavailable, the code falls back to CPU and exposes that fact through the system-status endpoint.

### 6.2 Image inspection workflow

The main endpoint is:

```text
POST /api/v1/inspect
```

The workflow is:

1. Accept an uploaded JPG, PNG, or WEBP file.
2. Reject missing, empty, oversized, or invalid image files.
3. Sanitize the filename and save a temporary upload.
4. Validate optional confidence, IoU, and image-size parameters.
5. Run the shared inference service.
6. Convert model results into a stable API schema.
7. Draw boxes and segmentation masks on a copy of the image.
8. Save the original image, annotated image, and `result.json`.
9. Append a compact record to inspection history.
10. Delete the temporary upload after processing.

The model is guarded by a prediction lock so concurrent requests do not use the same model object unsafely.

### 6.3 Result and history APIs

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Basic service health |
| `GET /api/v1/system/status` | Backend, model, device, CUDA, GPU, and library status |
| `GET /api/v1/inspections` | Recent inspection history |
| `GET /api/v1/inspections/{id}` | One complete inspection result |
| `GET /api/v1/results/{id}` | Result JSON |
| `GET /api/v1/results/{id}/original` | Original stored image |
| `GET /api/v1/results/{id}/annotated` | Annotated image |

The result storage layout is human-readable and easy to inspect:

```text
backend/storage/inspections/<inspection-id>/
├── original.jpg
├── annotated.jpg
└── result.json
```

### 6.4 Security and validation already present

The backend includes several useful safeguards:

- file-size limits;
- image-format validation;
- filename sanitization;
- safe-path checks before serving stored results;
- parameter bounds for confidence, IoU, and image size;
- structured validation through Pydantic schemas; and
- explicit error responses for unavailable models and invalid requests.

These measures make the current local review system safer, but they are not a complete internet-facing security design. Authentication, authorization, rate limiting, encrypted transport, and production secret management are still needed for deployment beyond a trusted local network.

## 7. Implemented frontend

The dashboard is a Next.js 15 application under `frontend/`.

### 7.1 Pages and reviewer workflow

The implemented routes are:

- `/dashboard` — summary cards and recent inspections;
- `/inspection/new` — select an image and run inference;
- `/inspection/[id]` — original image, annotated image, detections, confidence, masks, and metadata;
- `/inspections` — searchable saved history;
- `/analytics` — live inspection totals and dataset context;
- `/experiments` — EXP001 and EXP002 review information;
- `/system` — simulator connection, mission status, telemetry, latest frame, and mission controls; and
- `/settings` — local API and appearance settings.

The frontend uses `frontend/lib/services.ts` as the API boundary and `frontend/lib/types.ts` for shared TypeScript contracts.

### 7.2 Live versus static information

This distinction is important in a project review:

- inspection history, results, images, backend status, and simulator state come from the FastAPI backend;
- experiment summaries and dataset-card values are review artifacts bundled in `frontend/lib/mock/demo-data.ts`;
- the dashboard does not fabricate simulator telemetry when the simulator is unavailable; and
- the browser falls back to an explicit unavailable/offline state when the backend is not running.

### 7.3 Simulation dashboard behavior

The System page subscribes to:

```text
ws://127.0.0.1:8000/api/v1/simulation/ws
```

It also performs periodic status refreshes. It can:

- show simulator readiness;
- request a mission start;
- request abort and return-to-launch;
- display mission states such as `ARMING`, `TAKEOFF`, `TRANSIT`, `INSPECTING`, `RETURNING`, `LANDING`, and `COMPLETED`;
- display MAVLink telemetry;
- show the latest simulated camera frame; and
- open the linked inspection result after a capture.

## 8. Implemented simulation

### 8.1 Simulation stack

The current implementation uses:

- Gazebo Harmonic for the virtual world and sensors;
- ArduPilot Copter SITL as the virtual flight controller;
- MAVLink over a local TCP connection;
- Python mission-control code in `simulation/mission/`;
- FastAPI as the Windows-side bridge and persistence layer; and
- Next.js as the review console.

The current simulation does not require ROS 2. Gazebo Transport is used directly for the camera subscription.

### 8.2 Simulated inspection world

The world file is:

```text
simulation/gazebo/worlds/aeroinspect_inspection.sdf
```

It contains:

- a ground plane;
- an Iris vehicle with a gimbal-mounted camera;
- an inspection wall model;
- lighting and physics;
- camera and sensor systems; and
- a Gazebo GUI configuration for a repeatable review view.

The camera is configured for a `640 × 480` frame, approximately 10 FPS, and a 70-degree horizontal field of view. The mission captures one settled frame at each inspection waypoint.

### 8.3 Mission controller

The main controller is:

```text
simulation/mission/mission_controller.py
```

It implements the following high-level sequence:

```text
Connect MAVLink and camera
        ↓
Register with FastAPI
        ↓
ARMING
        ↓
TAKEOFF to 3 m
        ↓
TRANSIT to the inspection approach point
        ↓
Visit five inspection waypoints
        ↓
At each point: settle → wait for a fresh camera frame → upload frame
        ↓
FastAPI runs inference and stores the linked result
        ↓
RETURNING / RTL
        ↓
LANDING
        ↓
COMPLETED after disarm confirmation
```

The mission coordinates are configured in `simulation/config/mission.yaml`, not scattered through the code. Coordinates are local NED for flight commands; Gazebo geometry is expressed in ENU.

### 8.4 Camera bridge

`simulation/mission/gazebo_camera.py` subscribes to the Gazebo image topic through Gazebo Transport. It:

- keeps the subscription objects alive;
- decodes RGB, grayscale, and RGBA payloads;
- converts to OpenCV BGR;
- resizes to the configured output size;
- encodes a JPEG; and
- waits for a fresh frame rather than accidentally reusing a stale frame.

Malformed frames are rejected and recorded in camera diagnostics. A missing fresh frame is reported as `CAMERA_UNAVAILABLE`.

### 8.5 FastAPI simulation bridge

Simulation endpoints are under `/api/v1/simulation`:

| Endpoint | Purpose |
|---|---|
| `GET /status` | Current connection, mission, telemetry, and capture state |
| `POST /session/connect` | Register the WSL controller and camera capability |
| `POST /mission/start` | Queue an autonomous mission |
| `POST /mission/stop` | Request abort/RTL |
| `POST /mission/state` | Publish controller state transitions |
| `POST /telemetry` | Publish the latest MAVLink snapshot |
| `POST /captures` | Upload a camera frame plus telemetry metadata and run inference |
| `GET /captures` | List saved simulation captures |
| `GET /frames/latest` | Serve the latest raw simulated camera frame |
| `WS /ws` | Stream simulation status and inspection events |

The capture endpoint performs inference in a worker thread so the FastAPI event loop can continue to accept telemetry and status requests while the model is running.

Each capture records:

- mission ID;
- frame ID;
- waypoint ID;
- camera topic and camera frame;
- frame dimensions and field of view;
- capture timestamp;
- telemetry snapshot; and
- the linked inspection result ID.

This is the foundation for answering the review question: “Where was this defect observed?”

## 9. Evidence that the simulation milestone has been reached

At the time of preparing this guide, `backend/storage/simulation/state.json` recorded:

- connection state: `READY` after completion;
- mission state: `COMPLETED`;
- five completed captures;
- a linked latest inspection ID;
- relative altitude back at zero;
- vehicle disarmed; and
- flight mode reported as RTL after confirmed landing.

The stored capture records show the configured camera topic, 640 × 480 frame size, waypoint IDs from `inspect-01` to `inspect-05`, live local NED telemetry, and linked annotated results.

This proves the software path has been exercised in simulation. It does **not** prove:

- that the real Raspberry Pi can run the model at the required frame rate;
- that the real camera is calibrated;
- that the real drone can safely fly the same waypoints;
- that the simulation model is physically accurate enough for flight certification; or
- that the model generalizes to every real structure, lighting condition, camera, or distance.

## 10. Verification performed for this review

The following checks were run against the current checkout:

| Check | Result |
|---|---|
| Backend pytest suite | **10 passed**, 2 dependency deprecation warnings |
| Frontend lint command (`tsc --noEmit`) | **Passed** |
| Frontend production build | **Passed** |
| Python compilation check for backend and mission modules | **Passed** |
| Saved simulation mission state | **Completed with 5 captures and confirmed landing** |

The automated backend tests cover health, system status, upload validation, inspection history, and safe behavior when the simulator is unavailable. The manual scripts in `tools/` provide additional running-backend smoke tests and a validation-split review flow.

## 11. What is implemented, partial, and left

### 11.1 Implemented now

- quality-checked crack segmentation dataset;
- trained YOLOv8n-Seg checkpoint;
- local CUDA/CPU inference service;
- upload validation and configurable thresholds;
- bounding boxes and segmentation masks;
- annotated image generation;
- persistent inspection history;
- FastAPI health, system, inspection, result, and simulation APIs;
- Next.js review dashboard;
- Gazebo inspection world;
- ArduPilot SITL MAVLink connection;
- autonomous simulated takeoff, waypoint inspection, RTL, and landing;
- direct Gazebo camera subscription;
- capture-to-inference-to-storage integration;
- linked telemetry and capture metadata; and
- WebSocket-based simulator status events.

### 11.2 Partially implemented

- **Real-time telemetry:** the simulator bridge and WebSocket exist, but the design is still local and not production-hardened.
- **Spatial defect localization:** the system stores the drone telemetry at capture time, but it does not yet project a 2D mask into accurate 3D wall coordinates.
- **Camera realism:** the current camera is simulated and the scene is deterministic. Lens distortion, calibration, exposure variation, motion blur, and real sensor noise are not fully modeled.
- **Severity:** detections are intentionally shown as “Not assessed.” No validated structural-severity classifier or engineering rule is implemented.
- **Frontend experiment data:** the review page contains curated experiment summaries rather than loading every metric dynamically from experiment files.
- **Deployment:** the system is suitable for local review; it is not yet packaged as a field-deployable companion-computer application.

### 11.3 Not implemented yet

- safe control of a physical Pixhawk-equipped drone;
- Raspberry Pi camera/MAVLink service;
- real-camera calibration and image-to-world scale estimation;
- onboard model acceleration or quantization for Raspberry Pi;
- robust Wi-Fi/telemetry link management;
- geofence, lost-link, low-battery, and operator-override policies for hardware;
- obstacle avoidance using the lab's optical-flow, lidar, or RealSense equipment;
- object tracking;
- path planning around arbitrary structures;
- repeated coverage planning for large structures;
- formal flight-test procedures and logs for the physical aircraft;
- authentication and encrypted transport for a networked deployment; and
- regulatory and operational approval for outdoor flights.

## 12. Attached laboratory material and how it relates to this project

The five attached images were used as context only. Their contents can be summarized as follows.

### 12.1 Available quadcopter reference

The quadcopter poster describes a micro S500 platform with approximately 100–200 g payload and about 15 minutes flight time. It lists a Pixhawk 2.4.8 controller, 2212/920 kV BLDC motors, 30 A ESCs with a 5 V/2 A BEC, 1045 propellers, an FS-i6 transmitter with FS-iA6B receiver, 433 MHz telemetry, optional SIYI A8 Mini camera, GPS, buzzer, FPV, optical flow, 12 m lidar, Raspberry Pi 3 companion computer, and a 3300 mAh 11.1 V 3S LiPo battery.

This is the closest poster match to the requested “pre-assembled drone with Raspberry Pi” integration path. The listed payload limit is important: the camera mount, Raspberry Pi, regulator, wiring, and any added sensor must stay within the available payload and center-of-gravity limits.

### 12.2 Available octocopter reference

The octocopter poster describes a Tarot TL X8 micro/small platform with approximately 1–2 kg payload and about 15–20 minutes flight time. It lists a Pixhawk V6X, 320 kV BLDC motors, 30 A ESC/BEC units, 1555 propellers, 915 MHz high-power telemetry, a SIYI A8 Mini camera, GPS, buzzer, FPV, optical flow, 12 m lidar, Jetson Nano/Raspberry Pi 3 companion options, and a 16.8 V 6S battery specification.

The octocopter provides more payload headroom, but it is not automatically the better first integration platform. It has different motor/ESC/propeller and power-system assumptions, so its flight-controller parameters, wiring, battery monitoring, and failsafes must be validated separately.

### 12.3 Lab equipment and workstation

The equipment poster lists Raspberry Pi 3, ESP32, DJI O3 air unit, goggles, motion controller, FPV controller, RealSense depth camera, chargers, battery analyzers, a 4S LiPo, Neo 3 Pro GPS, and engineering tools. The workstation is listed as an Intel i7-12700 system with 64 GB RAM, GeForce RTX 3060, SSD/HDD storage, and Windows 11. MATLAB, Mission Planner, QGroundControl, and AirSim are listed as software/simulation resources.

The current repository uses FastAPI, Next.js, Gazebo Harmonic, ArduPilot SITL, and MAVLink. Therefore, AirSim and MATLAB/Simulink are lab options and possible future experiment tools, not dependencies of the current implementation.

### 12.4 Lab experiment list

The poster lists build/calibration, semi-autonomous mission planning, Raspberry Pi 4 companion control, object tracking, MATLAB/Simulink UAV simulation, Pixhawk HITL, obstacle avoidance, navigation, and path planning experiments.

For this repository, the most direct mapping is:

| Lab experiment | Relationship to AeroInspectAI |
|---|---|
| Quadcopter and radio calibration | Required before any physical flight test; not done by this software repository |
| Semi-autonomous mission planning | Simulated analogue exists; physical implementation remains |
| Raspberry Pi companion computer for Pixhawk | Next hardware-integration milestone |
| Object tracking | Separate future vision capability |
| UAV simulation | Current repository already has a Gazebo/ArduPilot simulation path |
| Simulink-HITL with Pixhawk | Optional future validation path, not current architecture |
| Obstacle avoidance | Future sensor/control integration |
| UAV navigation and path planning | Current mission is deterministic waypoint navigation; general planning remains |

The ADAPT Lab vision poster reinforces the broader goal of AI, computer vision, autonomous flight, disaster response, logistics, agriculture, surveillance, and environmental monitoring. It provides project motivation, not implementation evidence.

## 13. Recommended review explanation

For a project review, the clearest explanation is:

> “AeroInspectAI is an AI-assisted drone inspection pipeline. We trained a crack-segmentation model, exposed it through a validated FastAPI service, built a Next.js review console, and integrated the same inference path with a simulated ArduPilot/Gazebo inspection mission. The simulator can take off, visit five configured inspection positions, obtain camera frames, run crack inference, save results with telemetry, and return to land. The remaining work is the transition from deterministic simulation to calibrated physical sensing, Raspberry Pi companion software, safe flight testing, and richer navigation/obstacle-avoidance behavior.”

That statement is accurate because it separates the current software demonstration from future physical-drone claims.

## 14. How to run the current software review

### 14.1 Backend

From the repository root, use the existing Python environment and start FastAPI from `backend/`:

```powershell
cd D:\Projects\AeroInspectAI\backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

The main checks are:

```text
http://127.0.0.1:8000/api/v1/health
http://127.0.0.1:8000/api/v1/system/status
http://127.0.0.1:8000/docs
```

### 14.2 Frontend

From `frontend/`:

```powershell
cd D:\Projects\AeroInspectAI\frontend
npm run dev
```

Open `http://localhost:3000`.

### 14.3 Simulation

The repository includes `START_SIMULATION.bat` and the WSL scripts under `simulation/scripts/`. The simulation must be started only after the backend is available, because the WSL controller registers with the FastAPI simulation bridge and uploads frames to it.

Before a review demonstration, confirm:

1. Backend health is green.
2. The model is loaded in System status.
3. The WSL simulation controller has registered as `READY`.
4. Gazebo and ArduPilot SITL are running.
5. The camera topic is producing fresh frames.
6. Start the mission from the System page.
7. Wait for five captures and confirmed landing.
8. Open the latest linked inspection result.

## 15. Real Raspberry Pi drone integration guide

This section describes a safe, staged path to connect the software to a pre-assembled Pixhawk drone with a Raspberry Pi companion computer. It is a design guide, not authorization to fly.

### 15.1 Recommended first target

Start with the listed Raspberry Pi quadcopter rather than the heavier octocopter, provided the final payload calculation is acceptable. The quadcopter is simpler and closer to the current simulated Iris mission. Confirm the actual installed hardware; a poster specification is not a substitute for inspecting the vehicle.

The physical system should be split into two control layers:

```text
Raspberry Pi companion computer
  - camera capture
  - image compression
  - inference request / optional local inference
  - mission-level commands
  - telemetry forwarding
              │ MAVLink
              ▼
Pixhawk flight controller
  - stabilization
  - motor outputs
  - arming checks
  - altitude/navigation control
  - RTL and failsafes
              │
              ▼
Motors, ESCs, GPS, lidar/optical flow, battery monitor
```

The Raspberry Pi must not replace the Pixhawk's real-time stabilization loop.

### 15.2 Physical integration components

| Function | Likely hardware from lab poster | Integration concern |
|---|---|---|
| Flight controller | Pixhawk 2.4.8 on the quad, or Pixhawk V6X on the octocopter | Confirm firmware, port labels, parameters, and connector wiring |
| Companion computer | Raspberry Pi 3 listed for both platforms | Pi 3 may be too slow for comfortable real-time YOLO segmentation; benchmark before promising onboard FPS |
| Camera | SIYI A8 Mini, FPV camera, or a Raspberry Pi/USB camera | Need a usable digital stream and a known camera-to-body transform |
| Telemetry | 433 MHz or 915 MHz radio plus Wi-Fi where available | Do not send large images over a low-bandwidth telemetry radio |
| Position | GPS, optional optical flow and lidar | Validate update rate, frame conventions, and indoor/outdoor behavior |
| Power | Dedicated regulated 5 V supply for Raspberry Pi | Do not power the Pi from an unverified rail; check current capacity and noise |
| Ground station | Windows workstation with backend/frontend | Receives images/results and provides operator control |

### 15.3 Electrical and mechanical checks

Before software integration:

1. Remove propellers for every bench test.
2. Identify the Pixhawk TELEM port intended for the companion computer.
3. Verify voltage, current capacity, connector pinout, and common ground with a multimeter.
4. Use a dedicated regulated 5 V supply for the Raspberry Pi if the existing BEC is not explicitly rated for the Pi and peripherals.
5. Mount the Pi, camera, regulator, and cables so that they do not interfere with propellers, GPS, compass, gimbal movement, or cooling.
6. Recalculate payload, center of gravity, current draw, and expected flight time.
7. Secure all connectors against vibration and accidental disconnect.

The software cannot compensate for an overloaded aircraft, a noisy power rail, a loose camera, or an incorrectly wired telemetry port.

### 15.4 Software to install on the Raspberry Pi

The Pi companion service should eventually include:

- a supported Raspberry Pi OS image;
- Python runtime;
- `pymavlink` or MAVSDK for MAVLink communication;
- OpenCV for camera acquisition and JPEG encoding;
- a small companion-controller service;
- a watchdog and automatic restart policy;
- a local queue for frames when the network is unavailable; and
- a configuration file for camera, backend, mission, and safety settings.

The current `simulation/mission/backend_client.py` is a useful starting point for the HTTP contract, but it must not be copied unchanged into a physical-flight controller.

### 15.5 Hardware-safe replacement for the simulation controller

The current simulation controller contains SITL-specific behavior, including a forced-arm magic value and assumptions about a local TCP connection. It must be isolated from the physical implementation.

For hardware, create a separate controller with these rules:

- default to telemetry read-only mode;
- require an explicit operator enable before sending movement or arm commands;
- never use the SITL force-arm value on a real vehicle;
- use normal Pixhawk pre-arm checks;
- allow the pilot to override or terminate the mission immediately;
- define maximum altitude, speed, distance, and inspection standoff;
- monitor battery, GPS quality, link health, and estimator status;
- configure and test RTL, geofence, and lost-link behavior in the flight controller;
- use a mission ID and monotonic frame IDs to prevent duplicate uploads; and
- stop image capture or return home when safety limits are reached.

### 15.6 Camera integration options

There are two common arrangements:

#### Option A — Raspberry Pi/USB camera

The Pi captures frames directly with OpenCV or a camera library. This is the simplest route for reusing the current HTTP upload contract. The camera must be rigidly mounted and calibrated.

#### Option B — SIYI A8 Mini or another gimbal camera

The Pi receives a digital stream from the camera or its companion link. The integration must document the actual video protocol, resolution, latency, gimbal orientation, and timestamp behavior. The SIYI camera's advertised 4K capability does not automatically mean that the Pi can decode, resize, infer, and transmit 4K frames in real time.

For the first field test, use a moderate resolution such as 640 × 480 or 640 × 640 and capture still frames at controlled waypoints. Benchmark latency before attempting continuous video inference.

### 15.7 Where inference should run

There are three possible placements:

| Placement | Advantages | Limitations |
|---|---|---|
| Ground workstation | Uses the existing FastAPI/model stack and RTX GPU | Requires a reliable high-bandwidth link; not autonomous when disconnected |
| Raspberry Pi 3 | Fully self-contained | YOLOv8n-Seg may be too slow and memory-constrained without optimization |
| Jetson Nano or stronger companion | Better fit for onboard CUDA inference | Adds a different deployment target and requires benchmarking/packaging |

The safest first physical integration is usually:

```text
Pi captures a frame → compresses it → sends it over Wi-Fi → workstation FastAPI runs inference
```

After the flight and camera pipeline are reliable, consider exporting/optimizing the model for onboard inference. Do not claim real-time onboard AI until measured on the actual companion computer.

### 15.8 Network and data-flow design

Use two separate communication ideas:

1. **MAVLink link:** small commands and telemetry between Pi and Pixhawk.
2. **Image/results link:** Wi-Fi or another high-bandwidth channel between Pi and the ground backend.

The low-bandwidth telemetry radio should not be treated as an image transport. The Pi should:

- timestamp every frame;
- attach mission ID, waypoint ID, camera orientation, and telemetry;
- compress frames to a size suitable for the link;
- queue frames locally if the backend is temporarily unavailable;
- retry uploads without creating duplicate inspection records; and
- report link and queue status to the operator.

### 15.9 Physical test stages

Use a staged validation ladder:

1. **Bench, props removed:** Pi boots, camera works, telemetry is visible, backend receives a frame.
2. **Bench command test, props removed:** verify mode changes and mission cancellation; keep arming disabled unless the lab procedure explicitly requires it.
3. **Powered restrained test:** confirm vibration, camera stability, regulator temperature, and link behavior.
4. **Manual outdoor flight:** no AI commands; validate GPS, radio, RTL, battery behavior, and pilot control.
5. **Read-only AI flight:** capture and annotate images while the pilot flies manually.
6. **Stationary hover capture:** test one inspection position and standoff distance.
7. **Supervised waypoint test:** use a small geofenced area and a human pilot ready to take over.
8. **Expanded inspection mission:** add more waypoints only after repeatable logs and safe abort behavior.

Every stage should retain Pixhawk logs, companion logs, image timestamps, inference results, and operator notes.

### 15.10 Calibration and validation needed before real inspection

The physical system must establish:

- camera intrinsics and distortion;
- camera-to-body and gimbal orientation;
- relationship between vehicle pose and image pixels;
- inspection standoff distance;
- usable image resolution at that distance;
- lighting and exposure limits;
- model latency and dropped-frame behavior;
- minimum confidence policy for human review; and
- repeatability across concrete types, colors, shadows, and crack widths.

If physical dimensions are required, a calibration target or range sensor is needed. The current dashboard correctly labels physical dimensions as unavailable without calibration.

## 16. Recommended remaining roadmap

### Phase A — Finish and document the simulation milestone

- capture a clean demonstration run;
- save the mission ID and five capture IDs;
- show the dashboard's live simulator page;
- show one linked annotated inspection;
- document the camera topic and coordinate convention; and
- keep the simulation artifacts reproducible.

### Phase B — Improve simulation validity

- add repeatable lighting and camera-noise variants;
- vary wall distance and yaw;
- test empty-scene and false-positive cases;
- compare simulation frames with real camera frames;
- add explicit mission-error demonstrations; and
- improve 2D-to-world localization.

### Phase C — Bench-integrate the Raspberry Pi

- establish safe power and MAVLink wiring;
- run read-only telemetry;
- capture camera frames;
- upload frames to the existing backend;
- measure end-to-end latency; and
- validate queue/retry behavior.

### Phase D — Supervised physical flight

- perform manual flights first;
- validate RTL, geofence, battery, and lost-link behavior;
- capture images without autonomous movement;
- then test one stationary inspection point; and
- expand only after review of logs and detections.

### Phase E — Advanced autonomy

- object tracking;
- obstacle avoidance using lidar/depth/optical flow;
- adaptive standoff control;
- coverage/path planning for large structures;
- onboard optimized inference; and
- formal field validation with representative structures.

## 17. Final review takeaway

The strongest current claim is that AeroInspectAI is an integrated AI inspection prototype with a verified local inference workflow and a working simulated autonomous inspection pipeline. The next engineering challenge is not simply “connect Raspberry Pi to drone.” It is to preserve the current data contract while adding real camera calibration, safe Pixhawk companion control, reliable networking, hardware failsafes, and measured onboard/ground inference performance.

That separation—AI inspection, flight control, simulation, and operator safety—should remain visible throughout the next stages of the project.

