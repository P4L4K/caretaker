# Video and Audio Prediction Model Integration Fix

## Problem Identified
The front-end was not properly connected to the prediction models for video and audio analysis:
- **Audio**: WebSocket connection was working correctly ✅
- **Video**: The video feed was displaying but NOT sending frames to the prediction models ❌

## Root Cause
The `connectVideoStream()` function in `dashboard.js` was using an EventSource to connect to `/video/stream` endpoint, but this endpoint expects the backend to capture frames from a camera. The front-end was displaying the user's webcam but never sending those frames for analysis.

## Solution Implemented

### Changes to `frontend/js/dashboard.js`

#### 1. Updated `connectVideoStream()` Function
**Before**: Used EventSource to listen to server-side camera stream
```javascript
function connectVideoStream() {
    const eventSource = new EventSource(`${API_BASE}/video/stream`);
    window.videoEventSource = eventSource;
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        document.getElementById('current-emotion').textContent = data.mood;
        updateEmotionChart(data.mood);
        if (data.fall_detected) showFallAlert(data.timestamp);
    };
}
```

**After**: Captures frames from the video element and sends them to `/video/analyze` endpoint
```javascript
function connectVideoStream() {
    const videoElement = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 640;
    canvas.height = 480;
    
    const analyzeInterval = setInterval(async () => {
        if (!videoStream) {
            clearInterval(analyzeInterval);
            return;
        }
        
        try {
            // Draw current video frame to canvas
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            
            // Convert canvas to blob and send to backend
            canvas.toBlob(async (blob) => {
                if (!blob) return;
                
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');
                
                const response = await fetch(`${API_BASE}/video/analyze`, {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    document.getElementById('current-emotion').textContent = data.mood || 'neutral';
                    updateEmotionChart(data.mood || 'neutral');
                    if (data.fall_detected) showFallAlert(data.timestamp);
                }
            }, 'image/jpeg', 0.8);
        } catch (error) {
            console.error('Frame analysis error:', error);
        }
    }, 1000); // Analyze every 1 second
    
    window.videoAnalyzeInterval = analyzeInterval;
}
```

#### 2. Updated `toggleVideo()` Function
Added proper status indicators and cleanup:
- Updates video status dot and text when starting/stopping
- Shows "Active - Analyzing" when running
- Shows "Stopped" when not running
- Properly cleans up the analysis interval

#### 3. Updated `handleLogout()` Function
Added cleanup for the video analysis interval to prevent memory leaks

## How It Works Now

### Video Analysis Flow
1. User clicks "Start Video" button
2. Browser requests webcam access
3. Video stream displays in the `<video>` element
4. Every 1 second:
   - Current video frame is captured to a canvas
   - Frame is converted to JPEG blob
   - Frame is sent to `/video/analyze` endpoint via POST request
   - Backend runs:
     - **Emotion Recognition** using DeepFace
     - **Fall Detection** using YOLOv8-Pose
   - Results are displayed in the UI:
     - Current emotion shown in overlay
     - Emotion chart updated
     - Fall alert shown if detected

### Audio Analysis Flow (Already Working)
1. WebSocket connection to `/ws/audio`
2. Backend captures audio from microphone
3. Audio is processed with:
   - Noise reduction
   - Noise gate with adaptive threshold
   - YAMNet embeddings extraction
   - Cough classification using trained model
4. Results streamed back to frontend:
   - Waveform visualization
   - Audio levels (RMS, dB)
   - Cough detection alerts

## Backend Endpoints Used

### Video Endpoints
- `POST /video/analyze` - Analyzes a single frame for emotion and fall detection
  - Input: Image file (JPEG/PNG)
  - Output: `{mood: string, fall_detected: boolean, timestamp: string}`

### Audio Endpoints
- `WebSocket /ws/audio` - Real-time audio streaming and analysis
  - Streams waveform data and prediction results
  - Saves detected coughs to `media/cough/` directory

## Models Integrated

### Video Models
1. **Emotion Recognition** (`models/video/emotion_recognition.py`)
   - Uses DeepFace library
   - Detects: angry, disgust, fear, happy, sad, surprise, neutral
   - Backend: OpenCV face detector

2. **Fall Detection** (`models/video/fall_detection.py`)
   - Uses YOLOv8-Pose (ultralytics)
   - Detects human pose keypoints
   - Analyzes torso angle and hip movement speed
   - Triggers alert on sudden horizontal position change

### Audio Models
1. **Cough Detection** (`models/audio/cough/`)
   - YAMNet embeddings + Custom classifier
   - Uses patient metadata (age, gender, respiratory condition)
   - Threshold-based classification
   - Saves audio clips of detected coughs

## Testing Instructions

1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```

2. Open the frontend:
   ```
   Open frontend/dashboard.html in a browser
   ```

3. Login with valid credentials

4. Test Audio Monitoring:
   - Navigate to "Audio Monitoring" section
   - Should auto-connect via WebSocket
   - Make coughing sounds to test detection

5. Test Video Monitoring:
   - Navigate to "Video Monitoring" section
   - Click "Start Video" button
   - Allow webcam access
   - Should see:
     - Live video feed
     - Current emotion displayed
     - Emotion chart updating
     - Fall alert if you simulate a fall

## Performance Notes

- Video frames analyzed every 1 second (adjustable in code)
- Frame resolution: 640x480 (adjustable)
- JPEG quality: 80% (adjustable)
- Audio analysis: Real-time streaming with adaptive noise gate

## Files Modified

1. `frontend/js/dashboard.js`
   - `connectVideoStream()` - Complete rewrite
   - `toggleVideo()` - Added status indicators and cleanup
   - `handleLogout()` - Added interval cleanup

## Dependencies

All required dependencies are already installed:
- **Backend**: FastAPI, OpenCV, DeepFace, ultralytics, TensorFlow, librosa
- **Frontend**: Vanilla JavaScript, Chart.js, Font Awesome

## Status
✅ **FIXED** - Both video and audio prediction models are now properly integrated with the front-end
