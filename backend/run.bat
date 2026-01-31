@echo off
echo ============================================
echo   SmartLib Kiosk - Backend Server
echo ============================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Force correct model paths to override any stale environment variables
set FACE_MODEL_PATH=models/face_recognition/arcface_r100.onnx
set ANTISPOOFING_MODEL_PATH=models/anti_spoofing/minifasnet.onnx
set YOLO_MODEL_PATH=models/book_detection/yolov8m_books.pt

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt -q

REM Create logs directory
if not exist "logs" mkdir logs

REM Run the server
echo.
echo Starting SmartLib API Server...
echo API Docs: http://localhost:8000/docs
echo.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
