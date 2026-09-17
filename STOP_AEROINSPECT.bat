@echo off
setlocal
title AeroInspect AI Stopper
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_aeroinspect.ps1"
if errorlevel 1 pause
endlocal
