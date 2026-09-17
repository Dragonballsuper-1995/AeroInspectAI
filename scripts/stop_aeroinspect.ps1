[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $projectRoot ".aeroinspect"
$stopped = 0

foreach ($service in @("backend", "frontend")) {
    $pidFile = Join-Path $runtimeDir "$service.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) { continue }
    $servicePid = [int](Get-Content -LiteralPath $pidFile -Raw)
    $process = Get-Process -Id $servicePid -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $servicePid
        Write-Host "[STOPPED] $service (PID $servicePid)" -ForegroundColor Yellow
        $stopped += 1
    }
    Remove-Item -LiteralPath $pidFile -Force
}

if ($stopped -eq 0) {
    Write-Host "No launcher-managed AeroInspect processes were running."
}

