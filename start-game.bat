@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul || (
  echo uv was not found in PATH.
  echo Install uv, reopen the terminal, and run this file again.
  pause
  exit /b 1
)
uv sync --extra game || goto :failed
uv run --no-sync minoflux-game || goto :failed
exit /b 0
:failed
echo MinoFlux startup failed.
pause
exit /b 1
