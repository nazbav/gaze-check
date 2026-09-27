@echo off
cd /d "%~dp0"
chcp 65001 >nul
.venv\Scripts\python -m gaze_check %*
if errorlevel 1 pause
