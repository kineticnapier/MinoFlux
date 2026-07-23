@echo off
setlocal
cd /d "%~dp0"
uv sync --extra game
if errorlevel 1 exit /b %errorlevel%
uv run --no-sync minoflux-versus %*
