"""Simulation APIs used by the WSL mission controller and Next.js console."""

from __future__ import annotations

import asyncio
import json
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.schemas.simulation import CaptureMetadata, MissionRequest, SimulationTelemetry, SimulatorRegistration
from app.services.simulation_service import simulation_service


router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])


@router.get("/status")
async def get_status():
    return simulation_service.get_status()


@router.post("/session/connect")
async def connect_simulator(registration: SimulatorRegistration):
    return simulation_service.register(registration)


@router.post("/mission/start")
async def start_mission(request: MissionRequest = MissionRequest()):
    try:
        return simulation_service.request_start(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/mission/stop")
async def stop_mission():
    return simulation_service.request_stop()


@router.post("/mission/state")
async def update_mission_state(state: str = Form(...), detail: str = Form(...), waypoint_id: str | None = Form(None)):
    try:
        return simulation_service.update_mission(state, detail, waypoint_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/telemetry")
async def update_telemetry(telemetry: SimulationTelemetry):
    return simulation_service.update_telemetry(telemetry.model_dump(mode="json"))


@router.post("/captures")
async def submit_capture(
    file: UploadFile = File(...),
    metadata: str = Form(...),
):
    try:
        parsed = CaptureMetadata.model_validate(json.loads(metadata))
        return await simulation_service.submit_capture(file, parsed)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="metadata must be valid JSON") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/captures")
async def list_captures():
    return simulation_service.list_captures()


@router.get("/frames/latest")
async def latest_frame():
    path = simulation_service.latest_frame
    if not path.exists():
        raise HTTPException(status_code=404, detail="CAMERA_UNAVAILABLE: no captured frame")
    return FileResponse(path, media_type="image/jpeg")


@router.websocket("/ws")
async def simulation_websocket(websocket: WebSocket):
    await websocket.accept()
    subscriber = simulation_service.subscribe()
    try:
        while True:
            try:
                event = await asyncio.to_thread(subscriber.get, True, 15)
            except Exception:
                await websocket.send_json({"type": "heartbeat"})
                continue
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        simulation_service.unsubscribe(subscriber)
