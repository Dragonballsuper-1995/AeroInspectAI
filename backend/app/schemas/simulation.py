"""Contracts for the Gazebo/ArduPilot simulation bridge."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


SimulationConnection = Literal[
    "SIMULATION_UNAVAILABLE", "AWAITING_SIMULATOR", "READY", "RUNNING", "ERROR"
]
MissionState = Literal[
    "IDLE", "QUEUED", "ARMING", "TAKEOFF", "TRANSIT", "INSPECTING",
    "RETURNING", "LANDING", "COMPLETED", "ABORTED", "ERROR"
]


class SimulationTelemetry(BaseModel):
    """One MAVLink telemetry snapshot associated with a simulated frame."""

    timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    relative_altitude_m: Optional[float] = None
    north_m: Optional[float] = None
    east_m: Optional[float] = None
    down_m: Optional[float] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    ground_speed_m_s: Optional[float] = None
    battery_percent: Optional[float] = None
    flight_mode: str = "UNKNOWN"
    armed: bool = False


class SimulationStatus(BaseModel):
    connection_state: SimulationConnection = "SIMULATION_UNAVAILABLE"
    mission_state: MissionState = "IDLE"
    mission_id: Optional[str] = None
    waypoint_id: Optional[str] = None
    detail: str = "Start the WSL simulation stack to connect Gazebo and ArduPilot."
    telemetry: Optional[SimulationTelemetry] = None
    latest_frame_url: Optional[str] = None
    latest_inspection_id: Optional[str] = None
    captures_completed: int = 0
    updated_at: str


class SimulatorRegistration(BaseModel):
    """Capabilities reported by the WSL mission controller."""

    camera_topic: str
    vehicle: str = "iris_with_gimbal"
    detail: str = "Gazebo, ArduPilot SITL, camera, and MAVLink are connected."


class MissionRequest(BaseModel):
    mission_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class CaptureMetadata(BaseModel):
    mission_id: str = Field(..., min_length=1, max_length=64)
    frame_id: str = Field(..., min_length=1, max_length=64)
    waypoint_id: str = Field(..., min_length=1, max_length=64)
    captured_at: str
    camera_topic: str
    camera_frame: str = "camera_optical_frame"
    image_width: int = Field(..., gt=0, le=4096)
    image_height: int = Field(..., gt=0, le=4096)
    horizontal_fov_deg: float = Field(..., gt=1, lt=179)
    camera_source: str = "gazebo_transport"
    source_asset: Optional[str] = None
    demo_disclosure: Optional[str] = None
    telemetry: SimulationTelemetry


class CaptureRecord(BaseModel):
    capture_id: str
    inspection_id: str
    metadata: CaptureMetadata
    frame_url: str
    annotated_url: str
    cracks_detected: int
    created_at: str
