@echo off
setlocal
title AeroInspect AI Launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_aeroinspect.ps1"
if errorlevel 1 (
  echo.
  echo AeroInspect AI could not be started. Review the message above.
  pause
)
endlocal
