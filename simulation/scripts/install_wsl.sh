#!/usr/bin/env bash
# Install only Linux-side simulator dependencies. Run with sudo/root in Ubuntu 24.04.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DEBIAN_FRONTEND=noninteractive

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this once as root: sudo bash simulation/scripts/install_wsl.sh" >&2
  exit 1
fi

# `wsl.exe --user root` does not set SUDO_USER. Choose the normal WSL account
# explicitly so the runtime files land where START_SIMULATION.bat will find them.
RUN_USER="${AEROINSPECT_WSL_USER:-${SUDO_USER:-}}"
if [[ -z "${RUN_USER}" || "${RUN_USER}" == "root" ]]; then
  RUN_USER="$(getent passwd 1000 | cut -d: -f1)"
fi
[[ -n "${RUN_USER}" ]] || { echo "Could not determine the non-root WSL user." >&2; exit 1; }
SIM_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)/aeroinspect-sim"

apt-get update
apt-get install -y ca-certificates curl gnupg git cmake ninja-build pkg-config \
  python3-venv python3-pip python3-dev build-essential rapidjson-dev mesa-utils \
  ccache gawk wget valgrind screen python3-pexpect python3-empy astyle libtool libtool-bin \
  libxml2-dev libxslt1-dev lcov gcovr libcsfml-dev libsfml-dev libfreetype6-dev \
  libportmidi-dev libwxgtk3.2-dev python3-wxgtk4.0 xterm xfonts-base ppp \
  libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl
if [[ ! -f /etc/apt/sources.list.d/gazebo-stable.list ]]; then
  curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" \
    > /etc/apt/sources.list.d/gazebo-stable.list
  apt-get update
fi
apt-get install -y gz-harmonic libgz-sim8-dev libgz-transport13-dev \
  python3-gz-transport13 python3-gz-msgs10

install -d -o "${RUN_USER}" -g "${RUN_USER}" "${SIM_HOME}"
run_as_user() { runuser -u "${RUN_USER}" -- "$@"; }

if [[ ! -d "${SIM_HOME}/ardupilot/.git" ]]; then
  run_as_user git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "${SIM_HOME}/ardupilot"
fi
if [[ ! -d "${SIM_HOME}/ardupilot_gazebo/.git" ]]; then
  run_as_user git clone https://github.com/ArduPilot/ardupilot_gazebo.git "${SIM_HOME}/ardupilot_gazebo"
fi

# ArduPilot's upstream prerequisite script intentionally rejects root, while
# this reproducible installer runs as root to avoid a terminal sudo-password
# prompt. The SITL package set above is its Ubuntu-Noble equivalent; no MCU or
# hardware toolchains are needed for this Gazebo-only project.
run_as_user bash -lc "cd '${SIM_HOME}/ardupilot' && ./waf configure --board sitl && ./waf copter"
run_as_user bash -lc "cd '${SIM_HOME}/ardupilot_gazebo' && cmake -B build -G Ninja && cmake --build build -j2"
# Reuse Ubuntu's Gazebo Transport and OpenCV bindings from the WSL package set.
run_as_user python3 -m venv --system-site-packages "${SIM_HOME}/venv"
run_as_user "${SIM_HOME}/venv/bin/pip" install --upgrade pip
run_as_user "${SIM_HOME}/venv/bin/pip" install -r "${REPO_ROOT}/simulation/requirements.txt" MAVProxy
run_as_user python3 "${REPO_ROOT}/simulation/scripts/prepare_assets.py"

GZ_VERSION="$(gz --versions 2>/dev/null | sed -n '2p')"
ARDUPILOT_VERSION="$(run_as_user git -C "${SIM_HOME}/ardupilot" rev-parse HEAD)"
ARDUPILOT_GAZEBO_VERSION="$(run_as_user git -C "${SIM_HOME}/ardupilot_gazebo" rev-parse HEAD)"
{
  echo "gz=${GZ_VERSION:-Harmonic}"
  echo "ardupilot=${ARDUPILOT_VERSION}"
  echo "ardupilot_gazebo=${ARDUPILOT_GAZEBO_VERSION}"
} > "${REPO_ROOT}/simulation/.installed-versions"
chown "${RUN_USER}:${RUN_USER}" "${REPO_ROOT}/simulation/.installed-versions" || true
echo "Installation complete. Run simulation/scripts/start_simulation.sh as ${RUN_USER}."
