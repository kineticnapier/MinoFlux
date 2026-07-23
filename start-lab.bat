@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo uv was not found in PATH.
  echo Install uv, reopen the terminal, and run this file again.
  pause
  exit /b 1
)
uv sync --extra ui || goto :failed
uv run --no-sync minoflux-lab || goto :failed
exit /b 0
:failed
echo MinoFlux Lab startup failed.
pause
exit /b 1
