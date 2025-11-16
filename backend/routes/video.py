from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import cv2
import json
import os
import time
import threading
from typing import Optional
from datetime import datetime

# Import your model functions/classes
from models.video.emotion_recognition import get_emotion_from_frame
try:
    from models.video.fall_detection import FallDetector
    _fall_detector = FallDetector()
    FALL_DETECTION_AVAILABLE = True
except Exception as e:
    print(f"Warning: Fall detection not available: {e}")
    _fall_detector = None
    FALL_DETECTION_AVAILABLE = False

# Router setup
router = APIRouter(prefix="/video", tags=["video"])
_fall_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fall_log.json"))

# ----------------------- CAMERA CAPTURE THREAD -----------------------
class _VideoCaptureThread:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index = index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise HTTPException(status_code=500, detail="Could not open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

_cap = _VideoCaptureThread()

# ----------------------- LOG FALL EVENTS -----------------------
def _append_fall_log(event: dict):
    """Append fall event to fall_log.json"""
    try:
        if not os.path.exists(_fall_log_path):
            with open(_fall_log_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(_fall_log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
        data.append(event)
        with open(_fall_log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing fall log: {e}")

# ----------------------- EMOTION DETECTION (IMAGE) -----------------------
@router.post("/emotion")
async def detect_emotion_image(file: UploadFile = File(...)):
    """Detect emotion from uploaded image"""
    try:
        data = await file.read()
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        dominant, conf, emotions = get_emotion_from_frame(
            frame,
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            target_width=640,
        )

        return JSONResponse({
            "dominant_emotion": dominant,
            "confidence": conf,
            "emotions": emotions,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------- ANALYZE SINGLE FRAME -----------------------
@router.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze uploaded frame for emotion and fall detection"""
    try:
        data = await file.read()
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Use the enhanced process_video_frame function from unified_stream
        from .unified_stream import process_video_frame
        import asyncio
        
        # Process the frame with our enhanced function
        result = await process_video_frame(frame)
        
        # Log fall detection if detected
        if result.get('fall_detected'):
            fall_res = {
                "fall_detected": True,
                "timestamp": datetime.utcnow().isoformat(),
                "angle": result.get('fall_angle', 0)
            }
            _append_fall_log(fall_res)
        
        return JSONResponse({
            "mood": result.get('emotion', 'neutral'),
            "emotion_confidence": result.get('emotion_confidence', 0.0),
            "fall_detected": bool(result.get('fall_detected', False)),
            "timestamp": datetime.utcnow().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------- STREAM VIDEO (REAL-TIME) -----------------------
@router.get("/stream")
async def stream_video(camera_index: int = 0, interval_ms: int = 300, frame_skip: int = 10):
    """Stream real-time emotion and fall detection data"""
    try:
        if not _cap.running:
            _cap.index = camera_index
            _cap.start()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def gen():
        i = 0
        last_dom: Optional[str] = None
        while True:
            frame = _cap.read()
            if frame is None:
                time.sleep(max(0.01, interval_ms / 1000.0))
                continue

            # Run emotion detection every few frames
            if i % frame_skip == 0:
                try:
                    dom, _, _ = get_emotion_from_frame(
                        frame,
                        detector_backend="opencv",
                        enforce_detection=False,
                        align=True,
                        target_width=640,
                    )
                    if dom:
                        last_dom = dom
                except Exception as e:
                    print(f"Emotion detection error: {e}")

            # Run fall detection on every frame
            fall_res = {"fall_detected": False, "timestamp": None}
            if FALL_DETECTION_AVAILABLE and _fall_detector:
                fall_res = _fall_detector.detect_fall(frame)
                if fall_res.get("fall_detected"):
                    _append_fall_log(fall_res)
                    print(f"[ALERT] Fall detected at {fall_res.get('timestamp')}")

            payload = {
                "mood": last_dom or "neutral",
                "fall_detected": bool(fall_res.get("fall_detected")),
                "timestamp": fall_res.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(max(0.01, interval_ms / 1000.0))
            i += 1

    return StreamingResponse(gen(), media_type="text/event-stream")

# ----------------------- FALL TEST ENDPOINT -----------------------
@router.post("/fall-test")
async def test_fall(file: UploadFile = File(...)):
    """Test fall detection with a single uploaded frame"""
    try:
        if not FALL_DETECTION_AVAILABLE or not _fall_detector:
            raise HTTPException(status_code=503, detail="Fall detection is not available. Please install ultralytics.")
        
        data = await file.read()
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        fall_res = _fall_detector.detect_fall(frame)
        return JSONResponse(fall_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
