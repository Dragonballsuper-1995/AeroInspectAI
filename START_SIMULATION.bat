@echo off
setlocal
echo ============================================================
echo AEROINSPECT AI - GAZEBO + ARDUPILOT SIMULATION
echo ============================================================
echo Starting WSL simulator. Keep this window open while flying.
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/Projects/AeroInspectAI && bash simulation/scripts/start_simulation.sh"
pause
