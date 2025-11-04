from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2

# Import your function from the module we created
from models.video.emotion_recognition import get_emotion_from_frame

router = APIRouter(prefix="/video", tags=["video"])

@router.post("/emotion")
async def detect_emotion_image(file: UploadFile = File(...)):
    """
    Accepts an image file (jpg/png), detects emotion, and returns JSON.
    """
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

        return JSONResponse(
            {
                "dominant_emotion": dominant,
                "confidence": conf,           # 0-100
                "emotions": emotions,         # dict or null
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))