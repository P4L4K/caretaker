@echo off
echo ========================================
echo Video Model Test Script
echo ========================================
echo.

echo [1/4] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)
echo.

echo [2/4] Checking backend dependencies...
cd backend
python -c "import deepface; print('✓ DeepFace installed')" 2>nul || echo ✗ DeepFace NOT installed - Run: pip install deepface
python -c "import cv2; print('✓ OpenCV installed')" 2>nul || echo ✗ OpenCV NOT installed - Run: pip install opencv-python
python -c "import ultralytics; print('✓ Ultralytics installed')" 2>nul || echo ✗ Ultralytics NOT installed - Run: pip install ultralytics
python -c "import fastapi; print('✓ FastAPI installed')" 2>nul || echo ✗ FastAPI NOT installed - Run: pip install fastapi
echo.

echo [3/4] Checking model files...
if exist "yolov8n-pose.pt" (
    echo ✓ YOLOv8 pose model found
) else (
    echo ✗ YOLOv8 pose model NOT found
    echo   Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
)
cd ..
if exist "models\video\emotion_recognition.py" (
    echo ✓ Emotion recognition model found
) else (
    echo ✗ Emotion recognition model NOT found
)
if exist "models\video\fall_detection.py" (
    echo ✓ Fall detection model found
) else (
    echo ✗ Fall detection model NOT found
)
echo.

echo [4/4] Testing backend endpoint...
echo Starting backend server...
echo.
echo ========================================
echo Backend will start now.
echo.
echo After backend starts:
echo 1. Open browser to: http://localhost:8000/docs
echo 2. Test /video/analyze endpoint
echo 3. Or open: frontend/test_video.html
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd backend
python main.py
