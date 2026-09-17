#!/usr/bin/env bash
# Start Gazebo, SITL, MAVProxy and the controller. Keep this terminal open.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_HOME="${HOME}/aeroinspect-sim"
LOG_DIR="${REPO_ROOT}/simulation/logs"
PID_FILE="${LOG_DIR}/aeroinspect-stack.pids"
mkdir -p "${LOG_DIR}"

export GZ_SIM_SYSTEM_PLUGIN_PATH="${SIM_HOME}/ardupilot_gazebo/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:${GZ_SIM_SYSTEM_PLUGIN_PATH}}"
export GZ_SIM_RESOURCE_PATH="${REPO_ROOT}/simulation/gazebo/models:${SIM_HOME}/ardupilot_gazebo/models${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
export PYTHONPATH="${REPO_ROOT}/simulation${PYTHONPATH:+:${PYTHONPATH}}"
# Use XWayland for Qt under WSLg. The simulation server, drone camera, and
# observer GUI all use Ogre2 so the GUI receives the same rendered scene as the
# camera. Software Mesa is explicit here because WSLg reports llvmpipe/COPY
# MODE on this host; it is slower but avoids an empty accelerated context.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
unset WAYLAND_DISPLAY
# Keep this stack separate from abandoned/default Gazebo Transport sessions.
# The server, GUI, ArduPilot plugin, and camera subscriber inherit it.
export GZ_PARTITION="aeroinspect_${USER}"
BACKEND_HOST="$(ip route show default | awk '/default/ {print $3; exit}')"
export AEROINSPECT_BACKEND_URL="${AEROINSPECT_BACKEND_URL:-http://${BACKEND_HOST}:8000}"

terminate_pid() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
}

persist_pids() {
  printf '%s\n' "${SERVER_PID:-}" "${GUI_PID:-}" "${SITL_PID:-}" "${CONTROLLER_PID:-}" "${LOG_TAIL_PID:-}" >"${PID_FILE}"
}

stop_previous_stack() {
  [[ -f "${PID_FILE}" ]] || return 0
  while IFS= read -r pid; do
    terminate_pid "${pid}"
  done <"${PID_FILE}"
  rm -f "${PID_FILE}"
}

cleanup() {
  # Stop in dependency order. Unlike `gz sim`'s wrapper process, the separate
  # server / GUI PIDs below are tracked and cannot be left behind on Ctrl+C.
  for pid in "${LOG_TAIL_PID:-}" "${CONTROLLER_PID:-}" "${SITL_PID:-}" "${GUI_PID:-}" "${SERVER_PID:-}"; do
    terminate_pid "${pid}"
  done
  rm -f "${PID_FILE}"
}
stop_previous_stack
trap cleanup EXIT INT TERM

curl --fail --silent --show-error "${AEROINSPECT_BACKEND_URL}/api/v1/health" >/dev/null || {
  echo "BACKEND_UNAVAILABLE: start FastAPI on Windows, then verify ${AEROINSPECT_BACKEND_URL}." >&2
  exit 1
}
[[ -x "${SIM_HOME}/venv/bin/python" ]] || { echo "Simulation is not installed. Run install_wsl.sh first." >&2; exit 1; }

WORLD="${REPO_ROOT}/simulation/gazebo/worlds/aeroinspect_inspection.sdf"

# Start the Gazebo server and GUI independently. Previously `gz sim`'s parent
# wrapper could vanish when its WSL terminal closed, leaving an orphaned server
# and an empty COPY MODE GUI. Independent PIDs make repeat launches reliable.
gz sim -r -s --render-engine-server ogre2 "${WORLD}" >"${LOG_DIR}/gazebo.log" 2>&1 & SERVER_PID=$!
persist_pids
for _ in $(seq 1 30); do
  kill -0 "${SERVER_PID}" 2>/dev/null || {
    echo "SIMULATION_UNAVAILABLE: Gazebo server exited. See ${LOG_DIR}/gazebo.log." >&2
    exit 1
  }
  gz topic -l 2>/dev/null | grep -q "/world/aeroinspect_inspection/" && break
  sleep 1
done
gz topic -l 2>/dev/null | grep -q "/world/aeroinspect_inspection/" || {
  echo "SIMULATION_UNAVAILABLE: Gazebo world did not become ready." >&2
  exit 1
}

# The observer GUI is deliberately independent: a WSLg rendering failure is
# logged in gui.log but cannot stop the server, SITL, live camera, or mission.
# Use the stock Harmonic configuration so its scene manager, camera controls,
# and entity tree all attach to the Ogre2 world correctly.
gz sim -g --render-engine-gui ogre2 \
  --gui-config /usr/share/gz/gz-sim8/gui/gui.config >"${LOG_DIR}/gui.log" 2>&1 & GUI_PID=$!
persist_pids
SITL_BINARY="${SIM_HOME}/ardupilot/build/sitl/bin/arducopter"
[[ -x "${SITL_BINARY}" ]] || {
  echo "SITL is not built. Run install_wsl.sh first." >&2
  exit 1
}
# Use the completed SITL binary directly. sim_vehicle.py performs a build/update
# before it opens MAVLink, which races the fixed controller delay on WSL.
"${SITL_BINARY}" --model JSON:127.0.0.1 --sim-address=127.0.0.1 --speedup 1 --slave 0 \
  --defaults "${SIM_HOME}/ardupilot_gazebo/config/gazebo-iris-gimbal.parm" >"${LOG_DIR}/sitl.log" 2>&1 & SITL_PID=$!
persist_pids
for _ in $(seq 1 30); do
  kill -0 "${SITL_PID}" 2>/dev/null || {
    echo "MAVLINK_CONNECTION_LOST: ArduPilot SITL exited. See ${LOG_DIR}/sitl.log." >&2
    exit 1
  }
  ss -ltn 2>/dev/null | grep -q ':5760' && break
  sleep 1
done
ss -ltn 2>/dev/null | grep -q ':5760' || {
  echo "MAVLINK_CONNECTION_LOST: ArduPilot did not open TCP port 5760." >&2
  exit 1
}
"${SIM_HOME}/venv/bin/python" -m mission.mission_controller >"${LOG_DIR}/controller.log" 2>&1 & CONTROLLER_PID=$!
# Keep launcher feedback live without making the controller a child of `tee`.
# This lets cleanup signal the real Python PID rather than leaving it orphaned.
tail -n 0 -f "${LOG_DIR}/controller.log" & LOG_TAIL_PID=$!
persist_pids

wait "${CONTROLLER_PID}"
