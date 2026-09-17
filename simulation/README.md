# AeroInspectAI simulation

This subsystem connects a Gazebo Harmonic Iris flight to the Windows FastAPI API. It reuses the deployed EXP001 YOLOv8n-Seg checkpoint; it does not train, substitute, or fabricate a model result.

## Architecture

`Gazebo / SITL telemetry → WSL controller (Gazebo Transport + pymavlink) → FastAPI /api/v1/simulation/captures → EXP001 inference → Next.js System page`

Gazebo uses ENU geometry. The controller uses local NED only: `NED (N,E,D) → ENU (E,N,-D)`. The concrete wall is at ENU `(10, 0, 2)`. Five image panels are copied deterministically from `datasets/aeroinspect_crack_v1/images/train/`; they are simulator demonstration assets, never evaluation data.

## One-time installation

Start the Windows backend first, then run the following in an elevated PowerShell window. The script installs Gazebo Harmonic, ArduPilot SITL, the official `ardupilot_gazebo` plugin, MAVProxy, and the WSL-only Python dependencies. It can take significant disk space and time.

```powershell
wsl.exe -d Ubuntu-24.04 --user root -- bash /mnt/d/Projects/AeroInspectAI/simulation/scripts/install_wsl.sh
```

The installed package/repository revisions are written to `simulation/.installed-versions` for the local machine. If Ubuntu 24.04 fails to build the direct plugin, use the documented Ubuntu 22.04 compatibility fallback rather than mixing Gazebo major versions.

## Start and run a mission

1. Start FastAPI and Next.js with `START_AEROINSPECT.bat`. The bundled launcher exposes FastAPI to the WSL virtual network while the browser still uses `127.0.0.1`.
2. Run `START_SIMULATION.bat` and wait for “MAVLink and camera connected”.
3. Open **System** in the dashboard; its simulator status becomes `READY`.
4. Select **Start inspection mission**. The controller arms, takes off to 3 m, visits five NED inspection points at `E=7`, captures one settled 640×480 JPEG per point, submits each to FastAPI, returns home, and confirms landing.

Every submitted capture is a fresh Gazebo Transport RGB frame from the drone's simulated camera. Five training-split crack images are used only as materials on physical panels mounted on the simulated wall; they are never uploaded directly as camera payloads. The rendered frame therefore includes the camera pose, wall geometry, lighting, and scene background. The held-out test split is never used.

The dashboard never substitutes fake telemetry. When the stack is unavailable it shows `SIMULATION_UNAVAILABLE`; MAVLink, camera, backend, and inference failures appear as their explicit error codes.

The launcher uses XWayland (`QT_QPA_PLATFORM=xcb`, `WAYLAND_DISPLAY` unset), Ogre2, and the stock Gazebo Harmonic GUI configuration. This avoids the blank viewport caused by the former incomplete Ogre1 GUI configuration. The System page also shows the latest rendered camera frame, live NED telemetry, mission state, and linked inspection record.

## Validation and troubleshooting

Before a custom mission, validate the stock vehicle with the official flow: launch a Gazebo Iris world, run `sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON`, then arm, guided takeoff, move, RTL, and land. Inspect `simulation/logs/gazebo.log`, `sitl.log`, `mavproxy.log`, and `controller.log` if the dashboard reports an error.

For WSLg rendering, verify `glxinfo -B` and `gz sim --versions`. On a mirrored-network configuration where Windows localhost is not reachable from WSL, configure WSL networking according to the ArduPilot WSL documentation before starting the stack.
