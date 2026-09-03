@echo off
setlocal
cd /d "%~dp0"
uv sync --extra game --extra ml
if errorlevel 1 exit /b %errorlevel%
uv run --no-sync minoflux-versus %*
