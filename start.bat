@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Setting up for the first time, this takes a few minutes...
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
  if not exist .venv\Scripts\python.exe (
    echo Python 3.10 or newer is required: https://www.python.org/downloads/
    pause
    exit /b 1
  )
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
)
.venv\Scripts\python server.py --open
pause
