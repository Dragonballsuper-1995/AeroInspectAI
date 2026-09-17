"""Save one real RGB frame from the running Gazebo camera topic.

This diagnostic deliberately never reads a dataset image. Run it from WSL:
``python -m scripts.capture_gazebo_frame /tmp/gazebo-camera.jpg``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from mission.gazebo_camera import GazeboCamera


TOPIC = "/world/aeroinspect_inspection/model/aeroinspect_iris/model/gimbal/link/pitch_link/sensor/camera/image"


def main() -> None:
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/aeroinspect-gazebo-camera.jpg")
    camera = GazeboCamera(TOPIC, 640, 480)
    camera.start()
    frame = camera.wait_for_frame(time.time(), 15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(frame.jpeg)
    print(f"Saved real Gazebo camera frame to {destination}")


if __name__ == "__main__":
    main()
