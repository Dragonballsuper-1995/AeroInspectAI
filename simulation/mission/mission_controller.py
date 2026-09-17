"""Fly the fixed inspection route and link every settled frame to real inference.

Run from WSL with ``python3 -m mission.mission_controller``. This process never
generates telemetry or detections: MAVLink and Gazebo are the only sources.
"""

from __future__ import annotations

import math
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pymavlink import mavutil

from .backend_client import BackendClient
from .gazebo_camera import GazeboCamera


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "mission.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MissionController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        connection = config["connection"]
        camera_config = config["camera"]
        self.backend = BackendClient(os.environ.get("AEROINSPECT_BACKEND_URL", connection["backend_url"]))
        self.master = mavutil.mavlink_connection(connection["mavlink"], source_system=245)
        self.camera = GazeboCamera(
            camera_config["topic"],
            camera_config["output_width"],
            camera_config["output_height"],
        )
        self.telemetry: dict[str, Any] = {"timestamp": now(), "flight_mode": "UNKNOWN", "armed": False}
        self.last_telemetry_push = 0.0
        self.last_status_check = 0.0
        self.abort_requested = False
        self.running = True
        self.mission_id: str | None = None

    def stop(self, *_: object) -> None:
        self.running = False

    def connect(self) -> None:
        print("Waiting for ArduPilot MAVLink heartbeat…", flush=True)
        heartbeat = self.master.wait_heartbeat(timeout=60)
        if heartbeat is None:
            raise RuntimeError("MAVLINK_CONNECTION_LOST: no SITL heartbeat within 60 seconds")
        self.master.target_system = heartbeat.get_srcSystem()
        self.master.target_component = heartbeat.get_srcComponent()
        # JSON SITL only guarantees heartbeats until a GCS asks for the streams.
        # Request the measurements used by the dashboard and waypoint checks.
        for message_id in (
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,
        ):
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                200_000,  # 5 Hz
                0, 0, 0, 0, 0,
            )
        self.camera.start()
        # Windows may take a few seconds to accept the first WSL bridge
        # request while the model process is warming up. Keep the vehicle
        # connected and retry registration instead of terminating the whole
        # simulator on one transient HTTP timeout.
        last_error: Exception | None = None
        for attempt in range(1, 7):
            try:
                self.backend.register(self.config["camera"]["topic"])
                self.backend.state("IDLE", "Controller connected; waiting for dashboard mission request.")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"Backend connection delayed (attempt {attempt}/6): {exc}", file=sys.stderr, flush=True)
                time.sleep(min(2.0 * attempt, 8.0))
        if last_error is not None:
            raise RuntimeError(f"BACKEND_UNAVAILABLE: {last_error}") from last_error
        print("MAVLink and camera connected.", flush=True)

    def poll(self) -> None:
        """Drain MAVLink and publish measured telemetry at 5 Hz or slower."""
        while True:
            message = self.master.recv_match(blocking=False)
            if message is None:
                break
            kind = message.get_type()
            if kind == "BAD_DATA":
                continue
            self.telemetry["timestamp"] = now()
            if kind == "GLOBAL_POSITION_INT":
                self.telemetry.update({
                    "latitude": message.lat / 1e7,
                    "longitude": message.lon / 1e7,
                    "relative_altitude_m": message.relative_alt / 1000.0,
                    "north_m": message.vx / 100.0 if False else self.telemetry.get("north_m"),
                    "ground_speed_m_s": math.hypot(message.vx, message.vy) / 100.0,
                })
            elif kind == "LOCAL_POSITION_NED":
                self.telemetry.update({"north_m": message.x, "east_m": message.y, "down_m": message.z})
            elif kind == "ATTITUDE":
                self.telemetry.update({
                    "roll_deg": math.degrees(message.roll),
                    "pitch_deg": math.degrees(message.pitch),
                    "yaw_deg": (math.degrees(message.yaw) + 360) % 360,
                })
            elif kind == "SYS_STATUS":
                if message.battery_remaining >= 0:
                    self.telemetry["battery_percent"] = float(message.battery_remaining)
            elif kind == "HEARTBEAT":
                self.telemetry["flight_mode"] = mavutil.mode_string_v10(message)
                self.telemetry["armed"] = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        if time.monotonic() - self.last_telemetry_push >= 0.2:
            self.telemetry["timestamp"] = now()
            try:
                self.backend.telemetry(self.telemetry)
            except Exception as exc:
                # A transient Windows/WSL bridge delay must not stop the
                # aircraft or discard a camera frame. The next poll retries.
                print(f"Backend telemetry update delayed: {exc}", file=sys.stderr, flush=True)
            self.last_telemetry_push = time.monotonic()

    def wait_until(self, predicate, timeout: float, failure: str) -> None:
        deadline = time.monotonic() + timeout
        while self.running and time.monotonic() < deadline:
            self.poll()
            self._check_abort()
            if predicate():
                return
            time.sleep(0.05)
        raise RuntimeError(failure)

    def _check_abort(self) -> None:
        if self.mission_id and time.monotonic() - self.last_status_check >= 1.0:
            try:
                self.abort_requested = self.backend.status().get("mission_state") == "ABORTED"
            except Exception as exc:
                print(f"Backend status poll delayed: {exc}", file=sys.stderr, flush=True)
            self.last_status_check = time.monotonic()
        if self.abort_requested:
            self.abort_requested = False
            self.return_to_launch("Abort requested from dashboard")
            raise RuntimeError("Mission aborted")

    def _set_mode(self, mode: str) -> None:
        mapping = self.master.mode_mapping()
        if mode not in mapping:
            raise RuntimeError(f"MAVLINK_CONNECTION_LOST: SITL does not expose {mode} mode")
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mapping[mode],
        )
        self.wait_until(lambda: self.telemetry.get("flight_mode") == mode, 15, f"MAVLINK_CONNECTION_LOST: mode {mode} not accepted")

    def arm_and_takeoff(self, altitude: float) -> None:
        self.backend.state("ARMING", "Switching to GUIDED and arming SITL.")
        self._set_mode("GUIDED")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            # 21196 is ArduPilot's force-arm magic value.  This controller is
            # restricted to Copter SITL; it must never be reused for hardware.
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 21196, 0, 0, 0, 0, 0,
        )
        self.wait_until(lambda: bool(self.telemetry.get("armed")), 20, "MAVLINK_CONNECTION_LOST: vehicle did not arm")
        self.backend.state("TAKEOFF", f"Taking off to {altitude:.1f} m AGL.")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude,
        )
        self.wait_until(lambda: (self.telemetry.get("relative_altitude_m") or 0) >= altitude * 0.85, 60, "MAVLINK_CONNECTION_LOST: takeoff altitude was not reached")

    def goto(self, target: list[float], waypoint_id: str, detail: str) -> None:
        north, east, down = target
        self.backend.state("TRANSIT", detail, waypoint_id)
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000101111111000,
            north, east, down,
            0, 0, 0, 0, 0, 0,
            math.radians(self.config["mission"]["yaw_degrees"]), 0,
        )
        radius = self.config["mission"]["arrival_radius_m"]
        def arrived() -> bool:
            if any(self.telemetry.get(key) is None for key in ("north_m", "east_m", "down_m")):
                return False
            return math.dist([self.telemetry["north_m"], self.telemetry["east_m"], self.telemetry["down_m"]], target) <= radius
        self.wait_until(arrived, self.config["mission"]["arrival_timeout_seconds"], f"MAVLINK_CONNECTION_LOST: waypoint {waypoint_id} was not reached")

    def capture(self, waypoint_id: str, frame_number: int) -> None:
        self.backend.state("INSPECTING", f"Settling at {waypoint_id} before one camera capture.", waypoint_id)
        settled_since = time.time()
        self.wait_until(lambda: time.time() - settled_since >= self.config["mission"]["settle_seconds"], self.config["mission"]["settle_seconds"] + 5, "CAMERA_UNAVAILABLE: settle timer interrupted")
        frame = self.camera.wait_for_frame(settled_since, 10)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp.write(frame.jpeg)
            frame_path = Path(temp.name)
        try:
            metadata = {
                "mission_id": self.mission_id,
                "frame_id": f"frame-{frame_number:02d}",
                "waypoint_id": waypoint_id,
                "captured_at": datetime.fromtimestamp(frame.captured_at, timezone.utc).isoformat(),
                "camera_topic": self.config["camera"]["topic"],
                "camera_frame": self.config["camera"]["frame"],
                "image_width": self.config["camera"]["output_width"],
                "image_height": self.config["camera"]["output_height"],
                "horizontal_fov_deg": self.config["camera"]["horizontal_fov_degrees"],
                "camera_source": "gazebo_transport",
                "source_asset": None,
                "demo_disclosure": "Rendered by the simulated drone camera; crack photos are mounted on the virtual wall as scene materials.",
                "telemetry": self.telemetry,
            }
            result = self.backend.capture(frame_path, metadata)
            print(f"{waypoint_id}: saved capture {result['capture_id']} / inspection {result['inspection_id']}", flush=True)
        finally:
            frame_path.unlink(missing_ok=True)

    def return_to_launch(self, detail: str) -> None:
        self.backend.state("RETURNING", detail)
        self._set_mode("RTL")
        self.backend.state("LANDING", "RTL active; waiting for confirmed landing.")
        self.wait_until(lambda: not self.telemetry.get("armed", True), 120, "MAVLINK_CONNECTION_LOST: landing was not confirmed")

    def run_mission(self, mission_id: str) -> None:
        self.mission_id = mission_id
        mission = self.config["mission"]
        self.arm_and_takeoff(mission["takeoff_altitude_m"])
        self.goto(mission["approach_ned_m"], "approach", "Approaching the inspection wall.")
        for index, point in enumerate(mission["inspection_waypoints_ned_m"], start=1):
            waypoint_id = f"inspect-{index:02d}"
            self.goto(point, waypoint_id, f"Moving to inspection waypoint {index} of 5.")
            self.capture(waypoint_id, index)
        self.return_to_launch("All five frames inspected; returning home.")
        self.backend.state("COMPLETED", "Mission completed after confirmed landing.")

    def serve(self) -> None:
        self.connect()
        while self.running:
            self.poll()
            try:
                status = self.backend.status()
            except Exception as exc:
                # Status is advisory; MAVLink and the camera continue to run
                # while the Windows API bridge recovers.
                print(f"Backend status delayed: {exc}", file=sys.stderr, flush=True)
                time.sleep(1.0)
                continue
            if status.get("mission_state") == "QUEUED" and status.get("mission_id"):
                try:
                    self.run_mission(status["mission_id"])
                except Exception as exc:
                    if self.telemetry.get("armed"):
                        try:
                            self.return_to_launch("Mission failed; commanding RTL before reporting the error.")
                        except Exception as rtl_exc:
                            print(f"RTL after mission error failed: {rtl_exc}", file=sys.stderr, flush=True)
                    if str(exc) == "Mission aborted":
                        try:
                            self.backend.state("ABORTED", "Mission aborted and RTL requested.")
                        except Exception as state_exc:
                            print(f"Unable to publish ABORTED state: {state_exc}", file=sys.stderr, flush=True)
                    else:
                        try:
                            self.backend.state("ERROR", str(exc))
                        except Exception as state_exc:
                            print(f"Unable to publish ERROR state: {state_exc}", file=sys.stderr, flush=True)
                    print(f"Mission error: {exc}", file=sys.stderr, flush=True)
                finally:
                    self.mission_id = None
            time.sleep(0.1)


def main() -> None:
    with CONFIG.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    controller = MissionController(config)
    signal.signal(signal.SIGINT, controller.stop)
    signal.signal(signal.SIGTERM, controller.stop)
    controller.serve()


if __name__ == "__main__":
    main()
