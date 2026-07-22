@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul || (echo Python launcher was not found.& pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv || goto :failed
".venv\Scripts\python.exe" -m pip install -e ".[game]" || goto :failed
".venv\Scripts\python.exe" -m minoflux.game || goto :failed
exit /b 0
:failed
echo MinoFlux startup failed.
pause
exit /b 1
