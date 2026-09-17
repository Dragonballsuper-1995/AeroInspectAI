# **AeroInspectAI — Simulation Implementation Specification**

### **Codex Source-of-Truth Document**

**Project:** AeroInspectAI  
**Component:** Autonomous Drone Inspection Simulation  
**Primary Simulator:** Gazebo Harmonic  
**Flight Controller:** ArduPilot SITL  
**Communication:** MAVLink  
**AI Model:** Model 7 — trained crack-segmentation model  
**Backend:** FastAPI  
**Frontend:** Next.js  
**Development Environment:** Windows 11 \+ WSL2 Ubuntu  
**GPU:** NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM  
**Project Root:** `D:\Projects\AeroInspectAI`

---

# **1\. Purpose**

AeroInspectAI is an AI-powered autonomous drone inspection system.

The simulation component must provide a realistic software-only environment in which a simulated quadcopter can:

1. Start in a virtual inspection environment.  
2. Take off autonomously.  
3. Navigate to predefined inspection locations.  
4. Fly near an inspection structure.  
5. Capture simulated camera images.  
6. Send those images to **Model 7**.  
7. Detect and segment cracks using the trained AI model.  
8. Associate detections with the drone's simulated position.  
9. Send inspection results to the FastAPI backend.  
10. Display the mission and inspection results through the Next.js dashboard.  
11. Complete the inspection mission and return/land safely.

The simulation is intended to demonstrate the complete AeroInspectAI pipeline without requiring physical drone hardware.

---

# **2\. Critical Development Constraint**

## **Start the simulation implementation from scratch.**

The simulation must be designed as a clean standalone subsystem for AeroInspectAI.

---

# **3\. Final System Architecture**

The target architecture is:

                        AEROINSPECTAI  
                              │  
                              │  
                    Mission / Inspection  
                         Controller  
                              │  
                              │ MAVLink  
                              ▼  
                    ┌───────────────────┐  
                    │   ArduPilot SITL 		 │  
                    │                   			 │  
                    │ Virtual Flight   			 │  
                    │ Controller        			 │  
                    └─────────┬─────────┘  
                                  │  
                                  │ simulated  
                                  │ vehicle state  
                                 ▼  
                    ┌───────────────────┐  
                    │ Gazebo Harmonic   		│  
                    │                   			│  
                    │ 3D World          		│  
                    │ Drone             			│  
                    │ Physics           			│  
                    │ Camera            		│  
                    │ IMU/GPS           		│  
                    └─────────┬─────────┘  
                              		│  
                              		│ Camera Frames  
                              		▼

                    ┌───────────────────┐  
                    │     Model 7       			│  
                    │                   			│  
                    │ Crack Detection   		│  
                    │ \+ Segmentation    		│  
                    └─────────┬─────────┘  
                              		│  
                              		│ Detection  
                              		│ \+ confidence  
                              		│ \+ mask  
                              		│ \+ location  
                              		▼  
                    ┌───────────────────┐  
                    │     FastAPI       			│  
                    │     Backend       		│  
                    │                  			│  
                    │ Mission API       		│  
                    │ Telemetry API     		│  
                    │ Detection API     		│  
                    │ Image API         		│  
                    └─────────┬─────────┘  
                              		│  
                              		│ HTTP/WebSocket  
                             		▼  
                    ┌───────────────────┐  
                    │     Next.js       			│  
                    │     Dashboard     		│  
                    │                   			│  
                    │ Mission Status    		│  
                    │ Drone Position    		│  
                    │ Live Image        		│  
                    │ Crack Detection   		│  
                    │ Inspection Report 		│  
                    └───────────────────┘  
---

# **4\. Responsibility of Each Component**

## **4.1 Gazebo Harmonic**

Gazebo is responsible for the simulated physical world.

It must provide:

* 3D environment  
* drone model  
* physics  
* gravity  
* collision  
* inspection structure  
* simulated camera  
* simulated GPS  
* simulated IMU  
* optional additional sensors later

Gazebo should be treated as the **world and physics simulator**.

It should not contain the main AeroInspectAI application logic.

---

# **5\. ArduPilot SITL**

ArduPilot SITL is the simulated flight controller.

It is responsible for:

* flight-control logic  
* arming  
* takeoff  
* altitude control  
* attitude control  
* navigation  
* waypoint execution  
* landing  
* flight modes  
* simulated telemetry

The system must use **Copter** SITL.

ArduPilot officially supports SITL without physical flight hardware, and SITL can be run from Linux/WSL2. ([ArduPilot](https://ardupilot.ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html?utm_source=chatgpt.com))

---

# **6\. Gazebo ↔ ArduPilot Communication**

The initial architecture should use:

Gazebo  
   ↕  
ArduPilot Gazebo Plugin  
   ↕  
ArduPilot SITL

The current ArduPilot documentation provides an official Gazebo integration using the `ardupilot_gazebo` plugin. The direct plugin does not require ROS/ROS 2\. ([ArduPilot.org](https://ardupilot.org/dev/docs/sitl-with-gazebo.html?utm_source=chatgpt.com))

Therefore:

## **ROS 2 is NOT required for Phase 1\.**

Do not introduce ROS 2 simply because Gazebo is commonly used with ROS.

ROS 2 may be considered later if the project genuinely requires it.

---

# **7\. Why Gazebo Instead of the Previous AirSim Plan?**

AirSim was considered during the earlier project planning.

However, the implementation should now use:

> **Gazebo Harmonic \+ ArduPilot SITL**

instead of making the original Microsoft AirSim repository the foundation.

The original AirSim repository itself announced that the original project would be archived and that development was moving toward Project AirSim. ([GitHub](https://github.com/microsoft/AirSim/blob/main/project_airsim.md?utm_source=chatgpt.com))

AirSim remains historically relevant and technically capable, including ArduPilot support and programmatic image/state APIs, but it should not be the primary simulator for this new implementation. ([GitHub](https://github.com/microsoft/AirSim?utm_source=chatgpt.com))

---

# **8\. Development Environment**

The host machine is:

Windows 11  
        │  
        └── WSL2  
             │  
             └── Ubuntu

The simulation stack should preferably run inside the Linux/WSL2 environment rather than attempting to mix Windows-native and Linux-native components unnecessarily.

ArduPilot officially documents SITL use through WSL2. ([ArduPilot.org](https://ardupilot.org/dev/docs/SITL-setup-landingpage.html?utm_source=chatgpt.com))

---

# **9\. Project Directory**

The existing project root is:

D:\\Projects\\AeroInspectAI

Windows accesses it as:

D:\\Projects\\AeroInspectAI

while WSL may access it as:

/mnt/d/Projects/AeroInspectAI

However, Codex should determine the best location for simulation source/build files.

## **Important**

Do not blindly put Linux build artifacts inside the Windows-mounted project directory if doing so creates performance or permission problems.

Prefer a clean Linux-side workspace for:

* Gazebo build files  
* ArduPilot build artifacts  
* SITL workspace  
* Gazebo plugin build artifacts

while keeping project source/configuration organized and accessible from the main AeroInspectAI project.

---

# **10\. Simulation Architecture**

The simulation should be implemented in phases.

---

## **Phase 0 — Environment Validation**

Before writing AeroInspectAI-specific code:

### **Verify:**

WSL2  
Ubuntu  
OpenGL hardware acceleration  
Gazebo Harmonic  
ArduPilot  
ArduPilot SITL  
MAVLink connectivity

Gazebo should first successfully launch independently.

Example validation:

gz sim \-v4 \-r shapes.sdf

This is the basic Gazebo installation test documented by ArduPilot. ([ArduPilot.org](https://ardupilot.org/dev/docs/sitl-with-gazebo.html?utm_source=chatgpt.com))

Do not proceed to custom AeroInspectAI functionality until this works.

---

# **11\. Phase 1 — Basic Drone Simulation**

First goal:

> Get a quadcopter flying in Gazebo under ArduPilot SITL control.

Use the official ArduPilot Gazebo Iris example as the initial reference.

The official documentation provides an Iris Gazebo example and corresponding ArduPilot SITL launch process. ([ArduPilot.org](https://ardupilot.org/dev/docs/sitl-with-gazebo.html?utm_source=chatgpt.com))

At this stage:

Gazebo  
  \+  
ArduPilot SITL  
  \+  
Iris quadcopter

must work.

The drone must be able to:

* start  
* arm  
* take off  
* hover  
* move  
* land

Do not integrate Model 7 yet.

---

# **12\. Phase 2 — Create AeroInspectAI Inspection Environment**

After basic flight works, create a custom Gazebo world.

The world should contain an inspection target.

Initial environment:

                Inspection Structure  
                       │  
                       │  
          ┌─────────────────────────┐  
          │                         │  
          │      CRACK TARGET       │  
          │                         │  
          │    ───────╱──────       │  
          │       ╲                │  
          │   ────────────         │  
          │                         │  
          └─────────────────────────┘  
                       ▲  
                       │  
                       │ camera  
                       │  
                   ┌───────┐  
                   │ DRONE │  
                   └───────┘

The structure can initially be:

* wall  
* concrete panel  
* bridge-like vertical surface  
* building façade

The first implementation should prioritize functionality rather than photorealism.

---

# **13\. Inspection Target**

The environment must contain visually identifiable crack patterns.

The initial target should be designed specifically for testing Model 7\.

Possible approach:

Concrete wall  
      \+  
crack texture/material  
      \+  
controlled lighting

The crack appearance should be visible from the drone camera.

The system should eventually support multiple crack locations.

Example:

Wall

┌───────────────────────────────┐  
│                               │  
│       ╲                       │  
│        ╲──────                │  
│              ╲               │  
│                               │  
│                ───╲           │  
│                    ╲───       │  
│                               │  
└───────────────────────────────┘  
---

# **14\. Phase 3 — Simulated Camera**

Attach a camera to the drone.

The camera should provide:

RGB image

Initial target:

Resolution: configurable  
FPS: configurable

Do not optimize these values prematurely.

Start with a moderate resolution that allows real-time inference on the RTX 4050\.

The camera must have a clearly defined coordinate frame relative to the drone.

---

# **15\. Camera Data Pipeline**

The intended pipeline is:

Gazebo Camera  
      │  
      ▼  
RGB Frame  
      │  
      ▼  
Python Simulation/Inspection Service  
      │  
      ▼  
Model 7  
      │  
      ├── Bounding box  
      ├── Segmentation mask  
      ├── Confidence  
      └── Detection metadata  
      │  
      ▼  
FastAPI  
      │  
      ▼  
Next.js  
---

# **16\. Model 7**

The currently selected production model is:

> **Model 7**

Model 7 is the trained crack-segmentation model from the AeroInspectAI ML pipeline.

Do not create a second unrelated inference model.

The simulation must reuse the existing Model 7 inference interface wherever possible.

---

# **17\. Existing Model Information**

The trained project uses Ultralytics YOLO segmentation.

The baseline experiment used:

YOLOv8n-seg

with the final dataset created after dataset-quality validation.

The final dataset contains:

Train: 3717  
Validation: 199  
Test: 112  
Total: 4028

The baseline model achieved approximately:

Validation Mask mAP50: \~0.696  
Test Mask mAP50:       \~0.645

These numbers are development/evaluation results and should not be silently changed or presented as simulation accuracy.

The simulation is an **integration environment**, not a replacement for the held-out ML evaluation.

---

# **18\. Model Inference Requirements**

For every camera frame selected for inference, Model 7 should return:

timestamp  
frame\_id  
detections\[\]

Each detection should contain, where available:

class  
confidence  
bounding\_box  
segmentation\_mask

Example conceptual structure:

{  
  "frame\_id": 125,  
  "timestamp": 1726500000,  
  "detections": \[  
    {  
      "class": "crack",  
      "confidence": 0.87,  
      "bbox": \[120, 80, 340, 250\],  
      "mask": "..."  
    }  
  \]  
}

The exact existing Model 7 API should be inspected before creating a duplicate interface.

---

# **19\. Phase 4 — Autonomous Inspection Mission**

Once basic flight and camera operation work, implement an inspection mission.

The initial mission should be deterministic.

Example:

START  
  │  
  ▼  
ARM  
  │  
  ▼  
TAKEOFF  
  │  
  ▼  
MOVE TO INSPECTION START  
  │  
  ▼  
INSPECTION PASS 1  
  │  
  ├── Capture images  
  ├── Run Model 7  
  └── Record detections  
  │  
  ▼  
INSPECTION PASS 2  
  │  
  ├── Capture images  
  ├── Run Model 7  
  └── Record detections  
  │  
  ▼  
RETURN  
  │  
  ▼  
LAND  
  │  
  ▼  
MISSION COMPLETE  
---

# **20\. Mission Waypoints**

The initial mission should use predefined waypoints.

Example:

                WP3  
                  ●  
                  │  
                  │  
                  ● WP2  
                  │  
                  │  
                  ● WP1  
                  │  
                  │  
               START  
                  ●

The exact coordinates must be defined in the simulation configuration rather than hardcoded throughout the application.

Example conceptual configuration:

mission:  
  takeoff\_altitude: 5.0

  inspection\_waypoints:  
    \- x: ...  
      y: ...  
      z: ...

    \- x: ...  
      y: ...  
      z: ...

    \- x: ...  
      y: ...  
      z: ...

  return\_to\_home: true  
  land\_after\_mission: true  
---

# **21\. Mission Controller**

Create a dedicated mission-control layer.

Suggested responsibility:

MissionController

It should manage:

IDLE  
 ↓  
ARMING  
 ↓  
TAKEOFF  
 ↓  
TRANSIT  
 ↓  
INSPECTING  
 ↓  
RETURNING  
 ↓  
LANDING  
 ↓  
COMPLETED

Error states should also be represented.

Example:

ERROR  
ABORTED

Do not scatter mission state logic across multiple unrelated files.

---

# **22\. Telemetry**

The simulation should expose telemetry including:

latitude  
longitude  
altitude  
x  
y  
z  
roll  
pitch  
yaw  
velocity  
flight mode  
armed state  
battery  
mission state

Where a value is not meaningfully simulated, clearly document it rather than inventing it.

---

# **23\. MAVLink**

MAVLink should be treated as the communication protocol between the mission/control layer and ArduPilot.

Conceptually:

Mission Controller  
       │  
       │ MAVLink  
       ▼  
ArduPilot SITL  
       │  
       ▼  
Gazebo

The implementation should avoid unnecessary custom communication protocols between the flight-control components.

---

# **24\. Mission Controller vs FastAPI**

These are different responsibilities.

## **Mission Controller**

Responsible for:

* communicating with SITL  
* issuing flight commands  
* reading telemetry  
* managing inspection mission  
* collecting camera frames  
* triggering inference

## **FastAPI**

Responsible for:

* exposing application APIs  
* serving mission status  
* serving telemetry  
* receiving/storing detections  
* communicating with Next.js  
* providing inspection results

Do not make FastAPI directly responsible for low-level drone flight control.

---

# **25\. FastAPI Integration**

The existing FastAPI backend must remain the application backend.

Add simulation-specific endpoints rather than creating a second backend.

Potential API design:

POST /simulation/start  
POST /simulation/stop

GET /simulation/status

POST /simulation/mission/start  
POST /simulation/mission/stop

GET /simulation/telemetry

GET /simulation/detections

GET /simulation/frames/latest

The exact endpoint names should be checked against the existing backend before implementation.

Do not duplicate existing APIs.

---

# **26\. Next.js Integration**

The existing Next.js frontend must communicate with FastAPI.

The dashboard should eventually provide:

### **Mission Status**

Mission: INSPECTING  
Drone: AIRBORNE  
Mode: AUTO

### **Telemetry**

Altitude  
Position  
Velocity  
Heading  
Flight Mode

### **Camera**

Live or latest inspection frame.

### **AI Detection**

Example:

CRACK DETECTED

Confidence: 87%

Location:  
Inspection Point 3

### **Mission Progress**

✓ Takeoff  
✓ Transit  
● Inspection  
○ Return  
○ Landing  
---

# **27\. Important Frontend Rule**

Do not create mock simulation data once real simulation data is available.

The target architecture is:

Gazebo  
   ↓  
ArduPilot  
   ↓  
Mission Controller  
   ↓  
FastAPI  
   ↓  
Next.js

The dashboard must eventually display real telemetry and real Model 7 detections generated by the simulation.

---

# **28\. Live Updates**

For rapidly changing telemetry, prefer a real-time mechanism such as:

WebSocket

rather than repeatedly polling every value through HTTP.

Possible architecture:

ArduPilot  
    ↓  
Mission Controller  
    ↓  
FastAPI WebSocket  
    ↓  
Next.js

However, do not implement WebSockets before the basic REST integration works.

Build incrementally.

---

# **29\. Detection Storage**

Each detection should be associated with:

mission\_id  
frame\_id  
timestamp  
drone\_position  
camera\_position/orientation  
confidence  
bounding box  
segmentation mask

This allows the system to eventually answer:

> Where did the drone detect the crack?

---

# **30\. Spatial Association**

A major objective of the simulation is to connect:

AI Detection  
      \+  
Drone Pose  
      \+  
Camera Parameters

into:

Approximate physical inspection location

Initially this can be a simple association:

Detection  
    ↓  
Frame  
    ↓  
Drone pose at frame timestamp  
    ↓  
Inspection waypoint

Do NOT initially attempt advanced 3D reconstruction.

The first version should establish a reliable association between a detection and the drone's inspection position.

---

# **31\. Future Spatial Localization**

Later, the system can evolve toward:

Camera calibration  
        \+  
Depth / geometry  
        \+  
Drone pose  
        \+  
Camera orientation  
        \+  
Pixel coordinates  
        ↓  
3D crack position

This is explicitly a later phase.

Do not block the first working simulation on this.

---

# **32\. Inspection Logic**

The first inspection algorithm should be simple and deterministic.

Example:

Move to inspection waypoint  
        ↓  
Orient camera toward structure  
        ↓  
Capture frame  
        ↓  
Run Model 7  
        ↓  
Store result  
        ↓  
Move to next inspection position

The first objective is reliable end-to-end operation.

Sophisticated autonomous path planning can be added later.

---

# **33\. Camera Orientation**

The inspection camera should be mounted/oriented so that the inspection surface is visible.

The initial implementation should support a fixed camera orientation.

Later:

Drone movement  
\+  
gimbal/camera orientation  
\+  
surface geometry

can be made adaptive.

Do not implement a complicated gimbal system in Phase 1\.

---

# **34\. Simulation Configuration**

Simulation parameters should be externalized.

Create something conceptually similar to:

simulation/  
├── config/  
│   ├── simulation.yaml  
│   ├── mission.yaml  
│   └── camera.yaml

Example:

simulation:  
  world: aeroinspect\_inspection  
  vehicle: iris

camera:  
  width: 1280  
  height: 720  
  fps: 15

mission:  
  takeoff\_altitude: 5  
  inspection\_speed: 1.0

Exact values should be selected after performance testing.

---

# **35\. Recommended Project Structure**

Codex should adapt this to the existing repository rather than blindly creating duplicate folders.

Suggested structure:

AeroInspectAI/  
│  
├── backend/  
│  
├── frontend/  
│  
├── model/  
│  
├── simulation/  
│   │  
│   ├── README.md  
│   │  
│   ├── config/  
│   │   ├── simulation.yaml  
│   │   ├── mission.yaml  
│   │   └── camera.yaml  
│   │  
│   ├── gazebo/  
│   │   ├── worlds/  
│   │   ├── models/  
│   │   ├── plugins/  
│   │   └── materials/  
│   │  
│   ├── ardupilot/  
│   │  
│   ├── mission/  
│   │   ├── mission\_controller.py  
│   │   ├── telemetry.py  
│   │   └── mavlink\_interface.py  
│   │  
│   ├── perception/  
│   │   ├── camera.py  
│   │   ├── inference.py  
│   │   └── detection.py  
│   │  
│   ├── integration/  
│   │   └── backend\_client.py  
│   │  
│   └── scripts/  
│       ├── start\_simulation.sh  
│       ├── stop\_simulation.sh  
│       └── run\_mission.py  
│  
└── docs/

The actual repository should be inspected first.

---

# **36\. Do Not Duplicate Existing Functionality**

Before creating anything, Codex must inspect:

backend/  
frontend/  
Model 7 integration  
existing APIs  
existing configuration  
existing project structure

If functionality already exists, extend it.

Do not create:

backend2  
frontend2  
model7\_new  
simulation\_backend  
duplicate API

unless there is a concrete architectural reason.

---

# **37\. Error Handling**

The simulation must handle:

### **Gazebo unavailable**

SIMULATION\_UNAVAILABLE

### **SITL unavailable**

FLIGHT\_CONTROLLER\_UNAVAILABLE

### **MAVLink connection lost**

MAVLINK\_CONNECTION\_LOST

### **Camera unavailable**

CAMERA\_UNAVAILABLE

### **Model 7 inference failure**

INFERENCE\_ERROR

### **Backend unavailable**

BACKEND\_UNAVAILABLE

The dashboard should receive meaningful status information.

---

# **38\. Safety / Simulation Boundaries**

This is a simulation-only system.

No physical drone control should be implemented as part of this task.

The system must clearly distinguish:

SIMULATION

from:

REAL HARDWARE

No code should assume that a physical flight controller is connected.

---

# **39\. Logging**

Every major component should provide useful logs.

Example:

\[SIM\] Gazebo connected  
\[SITL\] ArduPilot connected  
\[MAVLINK\] Connection established  
\[MISSION\] Mission started  
\[MISSION\] Takeoff complete  
\[MISSION\] Inspection waypoint 1 reached  
\[CAMERA\] Frame captured  
\[MODEL7\] Crack detected confidence=0.87  
\[API\] Detection submitted  
\[MISSION\] Inspection complete  
\[MISSION\] Landing

Logs should make debugging easy for a beginner.

---

# **40\. Simulation Run Modes**

The project should eventually support:

## **Mode 1 — Basic Simulation**

Gazebo  
\+  
ArduPilot

No AI.

Used for testing flight.

---

## **Mode 2 — Inspection Simulation**

Gazebo  
\+  
ArduPilot  
\+  
Camera  
\+  
Mission Controller  
\+  
Model 7

Used for AI inspection.

---

## **Mode 3 — Full System**

Gazebo  
\+  
ArduPilot  
\+  
Mission Controller  
\+  
Model 7  
\+  
FastAPI  
\+  
Next.js

This is the final demonstration mode.

---

# **41\. Development Order**

Codex must follow this order.

### **Step 1**

Inspect repository.

### **Step 2**

Verify WSL2.

### **Step 3**

Install/verify Gazebo Harmonic.

### **Step 4**

Install/verify ArduPilot SITL.

### **Step 5**

Install/build the official ArduPilot Gazebo plugin.

### **Step 6**

Run official Iris example.

### **Step 7**

Verify:

ARM  
TAKEOFF  
MOVE  
LAND

### **Step 8**

Create AeroInspectAI Gazebo world.

### **Step 9**

Add inspection structure.

### **Step 10**

Add camera.

### **Step 11**

Capture camera frames.

### **Step 12**

Connect Model 7\.

### **Step 13**

Implement mission controller.

### **Step 14**

Implement inspection waypoint mission.

### **Step 15**

Send detections to FastAPI.

### **Step 16**

Expose telemetry.

### **Step 17**

Connect Next.js dashboard.

### **Step 18**

Implement real-time updates.

### **Step 19**

Run complete end-to-end mission.

### **Step 20**

Document everything.

---

# **42\. Definition of Done**

The simulation is considered successful when the following complete workflow works:

START SYSTEM  
     ↓  
Launch Gazebo  
     ↓  
Launch ArduPilot SITL  
     ↓  
Drone appears in world  
     ↓  
Mission Controller connects  
     ↓  
Drone arms  
     ↓  
Drone takes off  
     ↓  
Drone navigates to inspection area  
     ↓  
Camera observes inspection structure  
     ↓  
Camera captures image  
     ↓  
Model 7 processes image  
     ↓  
Crack detected  
     ↓  
Detection recorded  
     ↓  
Detection associated with drone position  
     ↓  
FastAPI receives result  
     ↓  
Next.js displays result  
     ↓  
Drone continues inspection  
     ↓  
Inspection complete  
     ↓  
Drone returns  
     ↓  
Drone lands  
     ↓  
Mission marked COMPLETE  
---

# **43\. Minimum Demonstration Scenario**

For the first successful demo, keep the environment simple.

The demo should contain:

1 drone  
1 inspection wall  
3–5 inspection waypoints  
1 simulated camera  
several visible crack patterns  
1 Model 7  
1 mission  
1 dashboard

The drone should fly a predictable inspection path.

The dashboard should show:

Drone status  
Mission progress  
Drone position  
Camera image  
Crack detection  
Confidence  
Inspection result  
---

# **44\. Performance Requirements**

The RTX 4050 Laptop GPU has 6 GB VRAM.

The simulation should therefore avoid unnecessarily heavy rendering or inference settings.

Performance should be measured for:

Gazebo FPS  
Camera FPS  
Model 7 inference latency  
Mission-controller latency  
CPU usage  
GPU usage  
VRAM usage

Do not optimize prematurely.

First achieve:

> Correctness → Stability → Integration → Performance

in that order.

---

# **45\. Model Inference Performance**

Camera FPS does not necessarily need to equal inference FPS.

For example:

Camera: 15 FPS

Inference:  
5–10 FPS

may be sufficient for the demonstration.

The system may initially process:

every Nth frame

rather than every frame.

This should be configurable.

---

# **46\. Avoid Overengineering**

Do NOT initially implement:

* SLAM  
* ROS 2  
* autonomous obstacle avoidance  
* LiDAR navigation  
* reinforcement learning  
* dynamic path planning  
* 3D reconstruction  
* multi-drone coordination  
* photogrammetry  
* physical drone integration  
* advanced gimbal control

These are future possibilities, not requirements for the first working simulation.

---

# **47\. Future Extensions**

Once the basic system works, possible extensions include:

Obstacle avoidance  
      ↓  
Adaptive inspection path  
      ↓  
Multiple structures  
      ↓  
3D crack localization  
      ↓  
Crack severity estimation  
      ↓  
Inspection heatmap  
      ↓  
Automated inspection report  
      ↓  
Multi-drone inspection

These should not be implemented until the core simulation is stable.

---

# **48\. Testing Strategy**

Each layer must be tested independently.

## **Test A — Gazebo**

Verify world launches.

## **Test B — SITL**

Verify ArduPilot starts.

## **Test C — Flight**

Verify:

arm  
takeoff  
move  
land

## **Test D — Camera**

Verify images are received.

## **Test E — Model 7**

Verify inference independently.

## **Test F — Mission Controller**

Verify waypoint sequence.

## **Test G — FastAPI**

Verify telemetry/detection APIs.

## **Test H — Frontend**

Verify dashboard displays data.

## **Test I — End-to-End**

Run the complete inspection mission.

---

# **49\. Reproducibility**

The simulation must be reproducible.

Document:

OS  
Ubuntu version  
Gazebo version  
ArduPilot version/commit  
Python version  
MAVLink library/version  
Model 7 path/version  
FastAPI version  
Next.js version

Use configuration files where appropriate.

Avoid undocumented manual changes.

---

# **50\. Documentation Requirements**

Create:

simulation/README.md

It must explain:

1. Architecture  
2. Prerequisites  
3. Installation  
4. Gazebo setup  
5. ArduPilot SITL setup  
6. Plugin setup  
7. Starting the simulator  
8. Starting the mission controller  
9. Starting Model 7  
10. Starting FastAPI  
11. Starting Next.js  
12. Running a complete mission  
13. Troubleshooting  
14. Project structure

A beginner should be able to follow the README and reproduce the simulation.

---

# **51\. Startup Experience**

Eventually, the goal should be to make running the complete system straightforward.

For example:

./simulation/scripts/start\_simulation.sh

and:

python simulation/scripts/run\_mission.py

The exact commands may differ based on the implementation.

Do not create a complex launcher until the individual components work.

---

# **52\. Troubleshooting Documentation**

Document common problems such as:

Gazebo does not start  
OpenGL acceleration unavailable  
SITL cannot connect  
MAVLink connection failed  
Camera frames unavailable  
Model 7 cannot load  
GPU unavailable  
FastAPI unavailable  
Frontend cannot connect

Each should include:

Symptom  
Cause  
Solution  
---

# **53\. Important Architectural Principle**

The simulation must remain modular.

The system should allow:

Gazebo  
       ↓  
ArduPilot  
       ↓  
Mission Controller  
       ↓  
Model 7  
       ↓  
FastAPI  
       ↓  
Next.js

to be independently replaced or upgraded.

For example, changing the Model 7 implementation should not require rewriting Gazebo.

Changing the frontend should not require changing MAVLink logic.

---

# **54\. Codex Working Rules**

Codex must follow these rules while implementing this document.

### **Rule 1**

**Inspect before modifying.**

Do not assume the current repository structure.

### **Rule 2**

**Do not overwrite working backend/frontend functionality.**

Extend it.

### **Rule 3**

**Do not reuse LAB-7/LAB-8 simulation code.**

The simulation starts fresh.

### **Rule 4**

**Do not introduce ROS 2 unless required.**

The first implementation should use the direct ArduPilot Gazebo integration.

### **Rule 5**

**Do not integrate Model 7 until the drone simulation itself works.**

### **Rule 6**

**Do not integrate the frontend until the backend API works.**

### **Rule 7**

**Every phase must have a test.**

### **Rule 8**

**Do not claim something works unless it has actually been tested.**

### **Rule 9**

When an installation/version issue occurs, check the current official documentation before choosing a workaround.

### **Rule 10**

Keep the implementation beginner-readable.

Prefer:

clear files  
clear functions  
clear logs  
clear configuration

over unnecessarily abstract frameworks.

---

# **55\. Official References**

Codex should use the following as authoritative references for the simulator stack:

* [ArduPilot — Using SITL with Gazebo](https://ardupilot.org/dev/docs/sitl-with-gazebo.html?utm_source=chatgpt.com) — official Gazebo \+ ArduPilot integration documentation. ([ArduPilot.org](https://ardupilot.org/dev/docs/sitl-with-gazebo.html?utm_source=chatgpt.com))  
* [ArduPilot — ROS 2 with Gazebo](https://ardupilot.ardupilot.org/dev/docs/ros2-gazebo.html?utm_source=chatgpt.com) — relevant if ROS 2 is introduced later. ([ArduPilot](https://ardupilot.ardupilot.org/dev/docs/ros2-gazebo.html?utm_source=chatgpt.com))  
* [ArduPilot — SITL Setup](https://ardupilot.org/dev/docs/SITL-setup-landingpage.html?utm_source=chatgpt.com) — Linux/Windows/WSL SITL setup references. ([ArduPilot.org](https://ardupilot.org/dev/docs/SITL-setup-landingpage.html?utm_source=chatgpt.com))  
* [ArduPilot — SITL Simulator](https://ardupilot.ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html?utm_source=chatgpt.com) — SITL architecture and purpose. ([ArduPilot](https://ardupilot.ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html?utm_source=chatgpt.com))  
* [Microsoft AirSim GitHub](https://github.com/microsoft/AirSim?utm_source=chatgpt.com) — historical AirSim reference only. ([GitHub](https://github.com/microsoft/AirSim?utm_source=chatgpt.com))  
* [Microsoft AirSim archival announcement](https://github.com/microsoft/AirSim/blob/main/project_airsim.md?utm_source=chatgpt.com) — explains the transition from the original AirSim project. ([GitHub](https://github.com/microsoft/AirSim/blob/main/project_airsim.md?utm_source=chatgpt.com))

---

# **56\. Final Target**

The final AeroInspectAI system should demonstrate:

> **A simulated autonomous drone performing an inspection mission in a Gazebo environment, controlled by ArduPilot SITL, capturing inspection imagery, running Model 7 crack segmentation, associating detected cracks with the drone's inspection position, sending the results through FastAPI, and displaying the complete mission and inspection results in the Next.js dashboard.**

The system should be **simulation-first, modular, reproducible, testable, and understandable**.

---

## **Codex Starting Instruction**

**Do not immediately start implementing everything.**

First:

1. Inspect the existing AeroInspectAI repository.  
2. Identify the existing frontend, backend, Model 7 integration, and current APIs.  
3. Identify the WSL2/Ubuntu environment.  
4. Check whether Gazebo and ArduPilot SITL are already installed.  
5. Report the current state.  
6. Produce a proposed implementation plan based on the repository.  
7. Only then begin Phase 0\.

**Do not ask me to redesign the architecture unless an existing repository constraint makes the architecture technically impossible.**

The architecture in this document is the intended target.