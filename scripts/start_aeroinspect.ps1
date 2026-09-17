[CmdletBinding()]
param([switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $projectRoot ".aeroinspect"
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$nextCli = Join-Path $frontendDir "node_modules\next\dist\bin\next"

function Test-HttpEndpoint {
    param([string]$Url, [string]$RequiredText = "")
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
        if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 400) { return $false }
        return -not $RequiredText -or $response.Content -match [regex]::Escape($RequiredText)
    } catch {
        return $false
    }
}

function Test-TcpPort {
    param([int]$Port)
    $connection = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Wait-ForEndpoint {
    param([string]$Url, [System.Diagnostics.Process]$Process, [int]$TimeoutSeconds = 90)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpEndpoint -Url $Url) { return $true }
        if ($Process.HasExited) { return $false }
        Start-Sleep -Milliseconds 750
    }
    return $false
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "AEROINSPECT AI - APPLICATION LAUNCHER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host ""

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$backendHealth = "http://127.0.0.1:8000/api/v1/health"
if (Test-HttpEndpoint -Url $backendHealth) {
    Write-Host "[OK] Backend is already healthy on port 8000." -ForegroundColor Green
} else {
    if (Test-TcpPort -Port 8000) {
        throw "Port 8000 is occupied by another application. Close it, then run this launcher again."
    }
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw "Python environment not found at $pythonExe. Create .venv and install backend/requirements.txt."
    }

    Write-Host "[START] Loading the FastAPI backend and trained checkpoint..."
    $backendProcess = Start-Process -FilePath $pythonExe `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000") `
        -WorkingDirectory $backendDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDir "backend.out.log") `
        -RedirectStandardError (Join-Path $runtimeDir "backend.err.log")
    Set-Content -LiteralPath (Join-Path $runtimeDir "backend.pid") -Value $backendProcess.Id
    if (-not (Wait-ForEndpoint -Url $backendHealth -Process $backendProcess)) {
        throw "Backend did not become healthy. See .aeroinspect\backend.err.log."
    }
    Write-Host "[OK] Backend ready on port 8000." -ForegroundColor Green
}

$frontendUrl = "http://127.0.0.1:3000"
if (Test-HttpEndpoint -Url $frontendUrl -RequiredText "AeroInspect") {
    Write-Host "[OK] Frontend is already running on port 3000." -ForegroundColor Green
} else {
    if (Test-TcpPort -Port 3000) {
        throw "Port 3000 is occupied by another application. Close it, then run this launcher again."
    }
    if (-not (Test-Path -LiteralPath $nextCli)) {
        throw "Frontend dependencies are missing. Run 'npm install' in the frontend folder first."
    }
    $nodeExe = (Get-Command node.exe -ErrorAction Stop).Source

    Write-Host "[START] Starting the Next.js frontend..."
    $frontendProcess = Start-Process -FilePath $nodeExe `
        -ArgumentList @($nextCli, "dev", "-H", "127.0.0.1", "-p", "3000") `
        -WorkingDirectory $frontendDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDir "frontend.out.log") `
        -RedirectStandardError (Join-Path $runtimeDir "frontend.err.log")
    Set-Content -LiteralPath (Join-Path $runtimeDir "frontend.pid") -Value $frontendProcess.Id
    if (-not (Wait-ForEndpoint -Url $frontendUrl -Process $frontendProcess -TimeoutSeconds 60)) {
        throw "Frontend did not become ready. See .aeroinspect\frontend.err.log."
    }
    Write-Host "[OK] Frontend ready on port 3000." -ForegroundColor Green
}

Write-Host ""
Write-Host "AeroInspect AI is ready:" -ForegroundColor Cyan
Write-Host "  Dashboard: $frontendUrl"
Write-Host "  API docs:  http://127.0.0.1:8000/docs"
Write-Host "  Stop:      Double-click STOP_AEROINSPECT.bat"
Write-Host ""
if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
