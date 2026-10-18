#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Setting up for the first time, this takes a few minutes..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
( sleep 2; (command -v open >/dev/null && open http://127.0.0.1:8000) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8000) || true ) &
exec .venv/bin/python server.py
