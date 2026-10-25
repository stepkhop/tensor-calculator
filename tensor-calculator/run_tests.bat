@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите run.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q pytest
".venv\Scripts\python.exe" -m pytest -v
pause
