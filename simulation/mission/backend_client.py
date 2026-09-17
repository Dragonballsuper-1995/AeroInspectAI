"""Small, explicit HTTP client for the Windows FastAPI simulation bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/") + "/api/v1/simulation"
        self.session = requests.Session()

    def _request(self, method: str, path: str, timeout: tuple[float, float] = (3.0, 15.0), **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(method, f"{self.base_url}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def register(self, camera_topic: str) -> dict[str, Any]:
        return self._request("POST", "/session/connect", json={
            "camera_topic": camera_topic,
            "vehicle": "aeroinspect_iris",
            "detail": "Gazebo camera and ArduPilot MAVLink are connected in WSL; the camera sensor supplies rendered simulated frames.",
        })

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status", timeout=(3.0, 10.0))

    def state(self, state: str, detail: str, waypoint_id: str | None = None) -> dict[str, Any]:
        data = {"state": state, "detail": detail}
        if waypoint_id:
            data["waypoint_id"] = waypoint_id
        return self._request("POST", "/mission/state", timeout=(3.0, 15.0), data=data)

    def telemetry(self, snapshot: dict[str, Any]) -> None:
        self._request("POST", "/telemetry", timeout=(2.0, 5.0), json=snapshot)

    def capture(self, frame: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        with frame.open("rb") as image:
            # Inference is asynchronous on the API, but the response still
            # waits for the inspection record. Allow a cold model and GPU
            # startup without aborting an otherwise valid flight.
            return self._request(
                "POST",
                "/captures",
                timeout=(5.0, 180.0),
                files={"file": (frame.name, image, "image/jpeg")},
                data={"metadata": __import__("json").dumps(metadata)},
            )
