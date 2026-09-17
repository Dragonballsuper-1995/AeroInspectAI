#!/usr/bin/env bash
# Start Gazebo, SITL, MAVProxy and the controller. Keep this terminal open.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_HOME="${HOME}/aeroinspect-sim"
LOG_DIR="${REPO_ROOT}/simulation/logs"
mkdir -p "${LOG_DIR}"

export GZ_SIM_SYSTEM_PLUGIN_PATH="${SIM_HOME}/ardupilot_gazebo/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:${GZ_SIM_SYSTEM_PLUGIN_PATH}}"
export GZ_SIM_RESOURCE_PATH="${REPO_ROOT}/simulation/gazebo/models:${SIM_HOME}/ardupilot_gazebo/models${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
export PYTHONPATH="${REPO_ROOT}/simulation${PYTHONPATH:+:${PYTHONPATH}}"
# Use XWayland for Qt under WSLg.  The Harmonic stock GUI and the camera sensor
# both use Ogre2; the previous custom Ogre1 viewport loaded the entity tree but
# left the scene pane blank on this machine.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
unset WAYLAND_DISPLAY
BACKEND_HOST="$(ip route show default | awk '/default/ {print $3; exit}')"
export AEROINSPECT_BACKEND_URL="${AEROINSPECT_BACKEND_URL:-http://${BACKEND_HOST}:8000}"

cleanup() {
  for pid in "${CONTROLLER_PID:-}" "${SITL_PID:-}" "${GAZEBO_PID:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

curl --fail --silent --show-error "${AEROINSPECT_BACKEND_URL}/api/v1/health" >/dev/null || {
  echo "BACKEND_UNAVAILABLE: start FastAPI on Windows, then verify ${AEROINSPECT_BACKEND_URL}." >&2
  exit 1
}
[[ -x "${SIM_HOME}/venv/bin/python" ]] || { echo "Simulation is not installed. Run install_wsl.sh first." >&2; exit 1; }

# The stock Harmonic GUI config contains the complete scene-manager and camera
# plugins. It has been verified under this WSLg installation; the abbreviated
# world GUI configuration did not render a viewport reliably.
gz sim -r --render-engine-server ogre2 --render-engine-gui ogre2 \
  --gui-config /usr/share/gz/gz-sim8/gui/gui.config \
  "${REPO_ROOT}/simulation/gazebo/worlds/aeroinspect_inspection.sdf" >"${LOG_DIR}/gazebo.log" 2>&1 & GAZEBO_PID=$!
sleep 8
SITL_BINARY="${SIM_HOME}/ardupilot/build/sitl/bin/arducopter"
[[ -x "${SITL_BINARY}" ]] || {
  echo "SITL is not built. Run install_wsl.sh first." >&2
  exit 1
}
# Use the completed SITL binary directly. sim_vehicle.py performs a build/update
# before it opens MAVLink, which races the fixed controller delay on WSL.
"${SITL_BINARY}" --model JSON:127.0.0.1 --sim-address=127.0.0.1 --speedup 1 --slave 0 \
  --defaults "${SIM_HOME}/ardupilot_gazebo/config/gazebo-iris-gimbal.parm" >"${LOG_DIR}/sitl.log" 2>&1 & SITL_PID=$!
sleep 4
"${SIM_HOME}/venv/bin/python" -m mission.mission_controller |& tee "${LOG_DIR}/controller.log" & CONTROLLER_PID=$!

wait "${CONTROLLER_PID}"
