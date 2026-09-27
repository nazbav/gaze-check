@echo off
cd /d "%~dp0"
chcp 65001 >nul
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv || goto fail
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-dev.txt || goto fail
.venv\Scripts\python -m pip install --no-deps mediapipe==1.0.1 || goto fail
.venv\Scripts\python packaging\get_models.py || goto fail
echo.
echo Готово. Запуск: start.bat
pause
exit /b 0
:fail
echo Установка не удалась.
pause
exit /b 1
