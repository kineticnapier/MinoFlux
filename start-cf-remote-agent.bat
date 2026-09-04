@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [MinoFlux CF Remote] .venv not found.
  echo Run: uv sync --extra game --extra ml
  pause
  exit /b 1
)

if "%MINOFLUX_CF_REMOTE_URL%"=="" (
  echo [MinoFlux CF Remote] MINOFLUX_CF_REMOTE_URL is not set.
  echo Example: setx MINOFLUX_CF_REMOTE_URL "https://minoflux-remote.YOUR_SUBDOMAIN.workers.dev"
  pause
  exit /b 1
)

if "%MINOFLUX_CF_AGENT_TOKEN%"=="" (
  echo [MinoFlux CF Remote] MINOFLUX_CF_AGENT_TOKEN is not set.
  echo Set it to the same value as the Worker AGENT_TOKEN secret.
  pause
  exit /b 1
)

echo [MinoFlux CF Remote] Starting Cloudflare command agent...
".venv\Scripts\python.exe" tools\cf_remote_agent.py
set CODE=%ERRORLEVEL%

echo.
echo [MinoFlux CF Remote] Agent exited with code %CODE%.
pause
exit /b %CODE%
