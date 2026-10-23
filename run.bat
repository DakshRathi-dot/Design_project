@echo off
cd /d "%~dp0"
python --version || (echo Python is not installed. Install Python 3.11 from python.org and tick "Add to PATH". & pause & exit /b)
if not exist venv (python -m venv venv)
call venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo. & echo Package install FAILED - see the error above. & pause & exit /b)
echo.
echo ============================================================
echo  Models load in the background (first run downloads ~1 GB).
echo  Wait for the line: [models] all done
echo  Then open  http://localhost:8000  in Chrome.
echo ============================================================
python -m uvicorn main:app --host 127.0.0.1 --port 8000
echo.
echo Server stopped. Error shown above.
pause
