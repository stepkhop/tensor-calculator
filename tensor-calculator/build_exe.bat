@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Сборка одного исполняемого файла dist\TensorCalculator.exe
if not exist ".venv\Scripts\python.exe" (
    echo Сначала запустите run.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q pyinstaller
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --onefile --windowed --hidden-import PIL._tkinter_finder --name TensorCalculator main.py
echo.
echo Готово: dist\TensorCalculator.exe
pause
