#!/bin/sh
# Запуск в Linux и macOS
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
    "$PY" -m venv .venv || exit 1
fi
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt || exit 1
exec .venv/bin/python main.py
