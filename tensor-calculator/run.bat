@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Тензорный калькулятор

rem Поиск Python: сначала py-лаунчер, затем python в PATH
set "PY="
py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY (
    python --version >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python не найден.
    echo Установите Python 3.10 или новее с https://www.python.org/downloads/
    echo В установщике отметьте "Add python.exe to PATH" и не снимайте флажок "tcl/tk and IDLE".
    goto fail
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" || (
    echo Нужен Python 3.8 или новее.
    goto fail
)
%PY% -c "import tkinter" 2>nul || (
    echo В установленном Python нет модуля tkinter.
    echo Переустановите Python с python.org, оставив флажок "tcl/tk and IDLE".
    goto fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Первый запуск: создается окружение .venv и ставится matplotlib, это займет минуту...
    %PY% -m venv .venv || goto fail
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || (
    echo Не удалось установить matplotlib. Проверьте подключение к интернету.
    goto fail
)
".venv\Scripts\python.exe" main.py || goto fail
exit /b 0

:fail
echo.
echo Запуск не удался. Пришлите текст из этого окна.
pause
exit /b 1
