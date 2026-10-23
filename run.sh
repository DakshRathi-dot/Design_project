#!/bin/bash
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
echo "Open http://localhost:8000 in Chrome"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
