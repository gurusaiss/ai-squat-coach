#!/usr/bin/env bash
# Convenience setup + run script. See README.md for details / credential options.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required (MediaPipe does not yet support 3.13)." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required and was not found on PATH." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtualenv..."
  python3.11 -m venv .venv
fi

source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r backend/requirements.txt

echo ""
echo "Starting server at http://localhost:8000"
echo "(Set GROK_API_KEY / GEMINI_API_KEY beforehand for real AI write-ups;"
echo " otherwise the app falls back to static template feedback. See README.md.)"
echo ""

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend
