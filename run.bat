@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
echo Open http://localhost:8000
uvicorn main:app --port 8000
