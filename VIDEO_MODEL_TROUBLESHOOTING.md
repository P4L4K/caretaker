# Video Model Troubleshooting Guide

## Issue: Video Model Not Working on Frontend Dashboard

### Recent Fixes Applied

1. **Added video readiness check** - Wait for video element to be ready before capturing frames
2. **Enhanced error logging** - Added console logs for debugging
3. **Fixed emotion chart** - Added missing "disgust" emotion (7 emotions total)
4. **Improved timing** - Increased interval to 1.5 seconds to reduce load
5. **Better error handling** - Separate try-catch blocks for fetch and canvas operations

### How to Diagnose the Problem

#### Step 1: Check Browser Console
Open the browser console (F12 → Console tab) and look for:

**Expected logs when video starts:**
```
Starting video analysis...
Video analysis result: {mood: "happy", fall_detected: false, timestamp: "..."}
```

**Common error messages:**
- `Video not ready yet, skipping frame...` - Video element not loaded (normal for first few seconds)
- `Failed to create blob from canvas` - Canvas drawing issue
- `Video analysis failed: 500` - Backend error
- `Fetch error: NetworkError` - Backend not running or CORS issue

#### Step 2: Check Backend Status
1. Verify backend is running:
   ```bash
   cd backend
   python main.py
   ```
   Should see: `Uvicorn running on http://0.0.0.0:8000`

2. Check backend logs for errors:
   - Look for model loading errors
   - Check for DeepFace or OpenCV errors
   - Verify YOLOv8 model loaded

#### Step 3: Test Backend Endpoint Directly

**Using curl (Windows PowerShell):**
```powershell
# Take a test image with your webcam first, save as test.jpg
curl -X POST http://localhost:8000/video/analyze -F "file=@test.jpg"
```

**Expected response:**
```json
{
  "mood": "neutral",
  "fall_detected": false,
  "timestamp": "2024-11-07 14:03:00"
}
```

**Using browser:**
1. Open: http://localhost:8000/docs
2. Find `/video/analyze` endpoint
3. Click "Try it out"
4. Upload a test image
5. Execute and check response

#### Step 4: Check Video Element
In browser console, run:
```javascript
const video = document.getElementById('video-feed');
console.log('Video ready state:', video.readyState);
console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
console.log('Video stream:', video.srcObject);
```

**Expected output:**
```
Video ready state: 4 (HAVE_ENOUGH_DATA)
Video dimensions: 640 x 480
Video stream: MediaStream {...}
```

#### Step 5: Manual Frame Capture Test
In browser console, run:
```javascript
const video = document.getElementById('video-feed');
const canvas = document.createElement('canvas');
canvas.width = 640;
canvas.height = 480;
const ctx = canvas.getContext('2d');
ctx.drawImage(video, 0, 0, 640, 480);
canvas.toBlob(async (blob) => {
    console.log('Blob created:', blob.size, 'bytes');
    const formData = new FormData();
    formData.append('file', blob, 'test.jpg');
    const response = await fetch('http://localhost:8000/video/analyze', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    console.log('Result:', data);
}, 'image/jpeg');
```

### Common Issues and Solutions

#### Issue 1: "Video not ready yet" keeps appearing
**Cause:** Video stream not properly initialized  
**Solution:**
1. Check webcam permissions in browser
2. Ensure video element has `autoplay` attribute
3. Wait longer before starting analysis (already implemented with 2s timeout)

#### Issue 2: Backend returns 500 error
**Cause:** Model loading failed or missing dependencies  
**Solution:**
```bash
cd backend
pip install deepface opencv-python ultralytics
```

Check backend logs for specific error:
- `No module named 'deepface'` → Install deepface
- `No module named 'ultralytics'` → Install ultralytics
- `Cannot find face` → Normal, means no face detected in frame

#### Issue 3: CORS error
**Cause:** Frontend and backend on different origins  
**Solution:** Already configured in `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Issue 4: Emotion always shows "neutral"
**Cause:** DeepFace not detecting faces  
**Solution:**
1. Ensure face is visible and well-lit
2. Face should be front-facing
3. Move closer to camera
4. Check backend logs for face detection errors

#### Issue 5: Video freezes or lags
**Cause:** Analysis too frequent or system overloaded  
**Solution:** Increase interval in `dashboard.js`:
```javascript
}, 2000); // Change from 1500 to 2000 (analyze every 2 seconds)
```

#### Issue 6: Fall detection not working
**Cause:** YOLOv8 model not loaded  
**Solution:**
1. Check if `backend/yolov8n-pose.pt` exists
2. Install ultralytics: `pip install ultralytics`
3. Check backend startup logs for fall detector initialization

### Verification Checklist

Before testing, ensure:
- [ ] Backend is running on port 8000
- [ ] Frontend can access backend (check CORS)
- [ ] Webcam permissions granted in browser
- [ ] DeepFace installed: `pip list | grep deepface`
- [ ] OpenCV installed: `pip list | grep opencv`
- [ ] Video element has valid stream
- [ ] Browser console shows no errors

### Testing Procedure

1. **Start backend:**
   ```bash
   cd backend
   python main.py
   ```

2. **Open frontend:**
   - Open `frontend/dashboard.html` in browser
   - Login with valid credentials

3. **Navigate to Video Monitoring:**
   - Click "Video Monitoring" in sidebar

4. **Start video:**
   - Click "Start Video" button
   - Allow webcam access
   - Status should show "Active - Analyzing"

5. **Check console logs:**
   - Should see "Starting video analysis..."
   - Should see "Video analysis result: {...}" every 1.5 seconds

6. **Verify emotion detection:**
   - Current emotion should update (not stuck on "--")
   - Try different facial expressions
   - Emotion chart should accumulate data

7. **Test fall detection:**
   - Simulate a fall (rapid position change)
   - Fall alert should appear

### Debug Mode

To enable more verbose logging, add this to the top of `connectVideoStream()`:
```javascript
const DEBUG = true;
if (DEBUG) {
    console.log('Video element:', videoElement);
    console.log('Video ready state:', videoElement.readyState);
    console.log('Video stream:', videoStream);
}
```

### Performance Monitoring

Check performance in browser:
1. Open DevTools → Performance tab
2. Start recording
3. Start video monitoring
4. Record for 30 seconds
5. Stop recording
6. Look for:
   - Frame drops
   - Long tasks
   - Memory leaks

### Backend Model Status Check

Add this endpoint to `backend/routes/video.py` for debugging:
```python
@router.get("/status")
async def video_status():
    return {
        "fall_detection_available": FALL_DETECTION_AVAILABLE,
        "emotion_recognition_available": True,
        "models_loaded": {
            "fall_detector": _fall_detector is not None,
            "deepface": "deepface" in sys.modules
        }
    }
```

Then check: http://localhost:8000/video/status

### Expected Behavior

**When working correctly:**
1. Video feed displays webcam
2. Status shows "Active - Analyzing"
3. Console logs appear every 1.5 seconds
4. Current emotion updates in real-time
5. Emotion chart accumulates data
6. No errors in console
7. Backend logs show successful predictions

### Still Not Working?

If none of the above helps:

1. **Check backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Verify model files exist:**
   ```bash
   ls models/video/
   ls backend/yolov8n-pose.pt
   ```

3. **Test with a static image:**
   - Use the `/video/emotion` endpoint with an uploaded image
   - If this works, issue is with frame capture

4. **Check browser compatibility:**
   - Use Chrome or Edge (best support)
   - Avoid Safari (limited WebRTC support)

5. **Review backend logs carefully:**
   - Look for import errors
   - Check for model loading failures
   - Verify no port conflicts

### Contact Information

If issue persists, provide:
- Browser console logs (full output)
- Backend logs (startup and during video analysis)
- Browser and OS version
- Python version and installed packages (`pip list`)
