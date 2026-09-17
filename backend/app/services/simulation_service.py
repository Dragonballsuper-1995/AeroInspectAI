"""Persistent simulation state and capture-to-inference integration."""

from __future__ import annotations

import json
import asyncio
import queue
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings
from app.core.logging_config import logger
from app.schemas.simulation import (
    CaptureMetadata,
    CaptureRecord,
    MissionRequest,
    SimulationStatus,
    SimulatorRegistration,
)
from app.services.inspection_service import InspectionService
from app.utils.file_utils import ensure_directory, load_json, save_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimulationService:
    """Own simulation state; Gazebo remains external to the FastAPI process."""

    def __init__(self) -> None:
        self.root = settings.get_storage_path() / "simulation"
        self.captures_dir = self.root / "captures"
        self.state_file = self.root / "state.json"
        self.latest_frame = self.root / "latest.jpg"
        self._lock = threading.RLock()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        ensure_directory(self.root)
        ensure_directory(self.captures_dir)
        self.inspection_service = InspectionService()
        if not self.state_file.exists():
            self._save_status(SimulationStatus(updated_at=_now()))

    def _read_status(self) -> SimulationStatus:
        with self._lock:
            try:
                return SimulationStatus.model_validate(load_json(self.state_file))
            except Exception:
                status = SimulationStatus(updated_at=_now())
                self._save_status(status)
                return status

    def _save_status(self, status: SimulationStatus) -> None:
        status.updated_at = _now()
        save_json(status.model_dump(mode="json"), self.state_file)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, "timestamp": _now(), "data": payload}
        with self._lock:
            for subscriber in list(self._subscribers):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    self._subscribers.remove(subscriber)

    def _update(self, event_type: str, **changes: Any) -> SimulationStatus:
        with self._lock:
            status = self._read_status()
            for key, value in changes.items():
                setattr(status, key, value)
            self._save_status(status)
            data = status.model_dump(mode="json")
        self._emit(event_type, data)
        return status

    def get_status(self) -> SimulationStatus:
        return self._read_status()

    def register(self, registration: SimulatorRegistration) -> SimulationStatus:
        return self._update(
            "simulator_connected",
            connection_state="READY",
            mission_state="IDLE",
            detail=registration.detail,
        )

    def request_start(self, request: MissionRequest) -> SimulationStatus:
        status = self._read_status()
        if status.connection_state not in {"READY", "RUNNING"}:
            raise RuntimeError("SIMULATION_UNAVAILABLE: launch the WSL simulation stack first")
        mission_id = request.mission_id or f"SIM-{datetime.now():%Y%m%d}-{uuid4().hex[:8].upper()}"
        return self._update(
            "mission_queued",
            connection_state="RUNNING",
            mission_state="QUEUED",
            mission_id=mission_id,
            waypoint_id=None,
            captures_completed=0,
            latest_inspection_id=None,
            detail="Mission queued; the WSL controller will arm the Iris vehicle.",
        )

    def request_stop(self) -> SimulationStatus:
        status = self._read_status()
        if status.mission_state in {"IDLE", "COMPLETED", "ABORTED", "ERROR"}:
            return status
        return self._update(
            "mission_abort_requested",
            mission_state="ABORTED",
            detail="Abort requested; controller will command RTL and landing.",
        )

    def update_mission(self, mission_state: str, detail: str, waypoint_id: str | None = None) -> SimulationStatus:
        status = self._read_status()
        if mission_state not in {"IDLE", "QUEUED", "ARMING", "TAKEOFF", "TRANSIT", "INSPECTING", "RETURNING", "LANDING", "COMPLETED", "ABORTED", "ERROR"}:
            raise ValueError("Invalid mission state")
        connection = "READY" if mission_state in {"COMPLETED", "ABORTED", "ERROR", "IDLE"} else "RUNNING"
        return self._update(
            "mission_status",
            connection_state=connection,
            mission_state=mission_state,
            waypoint_id=waypoint_id,
            detail=detail,
        )

    def update_telemetry(self, telemetry: dict[str, Any]) -> SimulationStatus:
        return self._update("telemetry", telemetry=telemetry)

    async def submit_capture(self, file: UploadFile, metadata: CaptureMetadata) -> CaptureRecord:
        capture_id = f"CAP-{uuid4().hex[:10].upper()}"
        capture_dir = self.captures_dir / capture_id
        ensure_directory(capture_dir)
        suffix = Path(file.filename or "frame.jpg").suffix.lower() or ".jpg"
        raw_frame = capture_dir / f"frame{suffix}"
        try:
            content = await file.read()
            if not content:
                raise ValueError("CAMERA_UNAVAILABLE: received an empty frame")
            raw_frame.write_bytes(content)
            if raw_frame.stat().st_size > settings.get_max_upload_bytes():
                raise ValueError("Frame exceeds configured upload limit")
            shutil.copy2(raw_frame, self.latest_frame)

            # Inference is CPU/GPU-bound and can take several seconds on the
            # first frame. Keep FastAPI's event loop free so the SITL controller
            # can continue posting telemetry and polling mission status while
            # this inspection runs.
            result = await asyncio.to_thread(
                self.inspection_service.process_inspection,
                raw_frame,
                original_filename=f"{metadata.mission_id}_{metadata.frame_id}{suffix}",
            )
            record = CaptureRecord(
                capture_id=capture_id,
                inspection_id=result.inspection_id,
                metadata=metadata,
                frame_url="/api/v1/simulation/frames/latest",
                annotated_url=result.image.annotated_url,
                cracks_detected=result.summary.cracks_detected,
                created_at=_now(),
            )
            save_json(record.model_dump(mode="json"), capture_dir / "capture.json")
            status = self._read_status()
            status.latest_frame_url = record.frame_url
            status.latest_inspection_id = record.inspection_id
            status.captures_completed += 1
            status.mission_state = "INSPECTING"
            status.waypoint_id = metadata.waypoint_id
            status.detail = f"Captured {metadata.frame_id}; inspection {record.inspection_id} completed."
            self._save_status(status)
            self._emit("inspection_completed", record.model_dump(mode="json"))
            return record
        except Exception as exc:
            logger.exception("[SIMULATION] Capture %s failed", capture_id)
            self._update("simulation_error", mission_state="ERROR", connection_state="ERROR", detail=f"INFERENCE_ERROR: {exc}")
            raise

    def list_captures(self) -> list[CaptureRecord]:
        records: list[CaptureRecord] = []
        for capture_file in sorted(self.captures_dir.glob("*/capture.json"), reverse=True):
            try:
                records.append(CaptureRecord.model_validate(load_json(capture_file)))
            except Exception:
                logger.warning("Ignoring unreadable simulation capture: %s", capture_file)
        return records

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
        with self._lock:
            self._subscribers.append(subscriber)
        subscriber.put({"type": "status", "timestamp": _now(), "data": self._read_status().model_dump(mode="json")})
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)


simulation_service = SimulationService()
