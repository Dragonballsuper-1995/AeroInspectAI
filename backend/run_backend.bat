@echo off
REM Start AeroInspectAI FastAPI Backend

cd /d "%~dp0"

REM Reuse an already healthy API instead of failing with WinError 10013.
powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/v1/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
    echo AeroInspect AI backend is already healthy on http://localhost:8000
    echo API Docs: http://localhost:8000/docs
    exit /b 0
)

echo.
echo ============================================================
echo AEROINSPECT AI - FASTAPI BACKEND LAUNCHER
echo ============================================================
echo.

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found
    echo Creating .env from .env.example
    copy ".env.example" ".env"
)

REM Activate virtual environment if it exists
if exist "..\.venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\.venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at ..\.venv
    echo Please create and activate a virtual environment first
    echo.
    echo Usage:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
)

REM Start the backend
echo Starting backend on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
