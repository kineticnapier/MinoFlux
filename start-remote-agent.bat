@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [MinoFlux Remote] .venv not found.
  echo Run: uv sync --extra game --extra ml
  pause
  exit /b 1
)

echo [MinoFlux Remote] Starting GitHub command agent...
".venv\Scripts\python.exe" tools\remote_agent.py
set CODE=%ERRORLEVEL%

echo.
echo [MinoFlux Remote] Agent exited with code %CODE%.
pause
exit /b %CODE%
