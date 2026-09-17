"""Gazebo Transport camera subscriber with a deliberate 640x480 JPEG boundary."""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


@dataclass
class CameraFrame:
    jpeg: bytes
    captured_at: float


class GazeboCamera:
    def __init__(
        self,
        topic: str,
        width: int,
        height: int,
    ) -> None:
        self.topic = topic
        self.width = width
        self.height = height
        self._latest: CameraFrame | None = None
        self._lock = threading.Lock()
        # pybind owns the Gazebo subscription, but keeping all three Python
        # objects alive is important: otherwise a temporary bound-method
        # wrapper can be collected while Gazebo still has the subscription.
        self._node: Any | None = None
        self._image_type: Any | None = None
        self._image_callback: Callable[[Any], None] = self._on_image
        self._frame_event = threading.Event()
        self._frame_count = 0
        self._last_received_at: float | None = None
        self._last_frame_error: str | None = None

    def start(self) -> None:
        transport_module = None
        image_module = None
        # Harmonic on the supported Ubuntu 24.04 setup ships Transport 13 and
        # Messages 10.  Prefer it explicitly: scanning newer ABI package names
        # can leave a partially initialised namespace when a package is absent.
        try:
            transport_module = importlib.import_module("gz.transport13")
            image_module = importlib.import_module("gz.msgs10.image_pb2")
        except ImportError:
            pass
        # Gazebo packages intentionally version Transport and Messages independently
        # (Harmonic on Ubuntu 24.04 ships transport13 with msgs10).
        if not image_module:
            for version in range(20, 7, -1):
                try:
                    image_module = importlib.import_module(f"gz.msgs{version}.image_pb2")
                    break
                except ImportError:
                    continue
        if not transport_module or not image_module:
            raise RuntimeError("CAMERA_UNAVAILABLE: Gazebo Python transport bindings are not installed")
        self._image_type = image_module.Image
        self._node = transport_module.Node()
        if not self._node.subscribe(self._image_type, self.topic, self._image_callback):
            raise RuntimeError(f"CAMERA_UNAVAILABLE: could not subscribe to {self.topic}")

    def _on_image(self, message) -> None:
        try:
            width = int(message.width)
            height = int(message.height)
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image dimensions {width}x{height}")
            pixels = np.frombuffer(message.data, dtype=np.uint8)
            pixel_count = width * height
            if len(pixels) == 0 or len(pixels) % pixel_count:
                raise ValueError(f"unexpected image payload size {len(pixels)} for {width}x{height}")
            channels = len(pixels) // pixel_count
            image = pixels.reshape((height, width, channels))
            if channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif channels == 3:
                # Gazebo camera sensors normally publish RGB8; OpenCV expects BGR.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                return
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                received_at = time.time()
                with self._lock:
                    self._latest = CameraFrame(encoded.tobytes(), received_at)
                    self._frame_count += 1
                    self._last_received_at = received_at
                    self._last_frame_error = None
                self._frame_event.set()
        except Exception as exc:
            # A malformed frame is not an inspection frame; retain the previous valid one,
            # but keep a concise diagnostic for a later CAMERA_UNAVAILABLE report.
            with self._lock:
                self._last_frame_error = f"{type(exc).__name__}: {exc}"

    def diagnostics(self) -> dict[str, Any]:
        """Return non-image camera health details for controller diagnostics/tests."""
        with self._lock:
            return {
                "topic": self.topic,
                "subscribed": self._node is not None and self._image_type is not None,
                "frames_received": self._frame_count,
                "last_received_at": self._last_received_at,
                "last_frame_error": self._last_frame_error,
            }

    def wait_for_frame(
        self,
        newer_than: float,
        timeout_seconds: float,
    ) -> CameraFrame:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self._lock:
                frame = self._latest
            if frame and frame.captured_at >= newer_than:
                return frame
            # Event-driven waiting avoids repeatedly reading a stale frame. A
            # short timeout closes the tiny clear/set race without delaying a
            # new capture by more than a quarter of a second.
            self._frame_event.clear()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._frame_event.wait(min(remaining, 0.25))
        with self._lock:
            last_error = self._last_frame_error
        suffix = f" (last decoder error: {last_error})" if last_error else ""
        raise RuntimeError(f"CAMERA_UNAVAILABLE: no fresh Gazebo camera frame{suffix}")
