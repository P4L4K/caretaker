# CareTaker AI Backend - API Documentation

## Overview
Complete AI-powered elderly care monitoring system with real-time audio cough detection and video emotion/fall detection.

## Base URL
```
http://localhost:8000
```

---

## Authentication Endpoints

### 1. Signup
**POST** `/signup`

Register a new caretaker account. Sends welcome email on successful registration.

**Request Body:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePassword123",
  "care_recipients": [
    {
      "name": "Jane Doe",
      "age": 75,
      "gender": "Female",
      "respiratory_condition_status": false
    }
  ]
}
```

**Response:**
```json
{
  "code": "200",
  "message": "Caretaker registered successfully"
}
```

---

### 2. Login
**POST** `/login`

Authenticate and receive JWT access token.

**Request Body:**
```json
{
  "username": "john_doe",
  "password": "SecurePassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

### 3. Logout
**POST** `/logout`

Invalidate the current access token.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "code": "200",
  "message": "Logout successful"
}
```

---

## Audio Monitoring Endpoints

### 4. Real-time Audio Stream (WebSocket)
**WS** `/ws/audio?token=<access_token>`

Real-time audio streaming with cough detection.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/audio?token=YOUR_TOKEN');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  // Waveform data for visualization
  if (data.waveform) {
    console.log('Audio levels:', data.rms, data.db);
    console.log('Gate open:', data.gate_open);
  }
  
  // Prediction results
  if (data.event === 'prediction') {
    console.log('Cough detected!', data);
    // data.timestamp - ISO timestamp
    // data.probability - 0.0 to 1.0
    // data.label - "Cough" or "Not Cough"
    // data.media_url - URL to saved audio file
  }
};
```

**Sent Data (Continuous):**
```json
{
  "waveform": [0.1, 0.2, ...],
  "rms": 2500.0,
  "db": -22.5,
  "gate_open": true,
  "gate_level": 1.0,
  "hold_active": true
}
```

**Received Data (On Cough Detection):**
```json
{
  "event": "prediction",
  "timestamp": "2025-11-06T10:30:45.123456Z",
  "probability": 0.87,
  "label": "Cough",
  "media_url": "/media/cough/cough_20251106T103045123456Z.wav"
}
```

---

### 5. Get Cough Detections
**GET** `/api/cough/detections`

Retrieve all saved cough detection events with metadata.

**Response:**
```json
{
  "items": [
    {
      "timestamp": "2025-11-06T10:30:45.123456Z",
      "probability": 0.87,
      "label": "Cough",
      "media_url": "/media/cough/cough_20251106T103045123456Z.wav",
      "username": "john_doe",
      "caretaker_id": 1,
      "recipient_id": 1,
      "age": 75,
      "gender": "Female",
      "respiratory_condition": false
    }
  ]
}
```

**Search/Filter (Frontend Implementation):**
- Filter by date range
- Filter by care recipient
- Sort by timestamp
- Play audio segments directly

---

## Video Monitoring Endpoints

### 6. Analyze Single Frame
**POST** `/video/analyze`

Analyze uploaded image for emotion and fall detection.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
```json
{
  "mood": "happy",
  "fall_detected": false,
  "timestamp": "2025-11-06 10:30:45"
}
```

---

### 7. Real-time Video Stream (Server-Sent Events)
**GET** `/video/stream?camera_index=0&interval_ms=300&frame_skip=10`

Real-time emotion and fall detection from webcam.

**Parameters:**
- `camera_index`: Camera device index (default: 0)
- `interval_ms`: Update interval in milliseconds (default: 300)
- `frame_skip`: Process every Nth frame (default: 10)

**Response (SSE Stream):**
```
data: {"mood": "happy", "fall_detected": false, "timestamp": "2025-11-06 10:30:45"}

data: {"mood": "neutral", "fall_detected": false, "timestamp": "2025-11-06 10:30:46"}

data: {"mood": "sad", "fall_detected": true, "timestamp": "2025-11-06 10:30:47"}
```

**JavaScript Example:**
```javascript
const eventSource = new EventSource(
  'http://localhost:8000/video/stream?camera_index=0'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Emotion:', data.mood);
  
  if (data.fall_detected) {
    alert('FALL DETECTED!');
  }
};
```

---

### 8. Detect Emotion from Image
**POST** `/video/emotion`

Detect emotion from uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.92
}
```

---

### 9. Test Fall Detection
**POST** `/video/fall-test`

Test fall detection with uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
```json
{
  "fall_detected": true,
  "timestamp": "2025-11-06 10:30:45",
  "confidence": 0.85
}
```

---

## Dashboard Implementation Guide

### Frontend Requirements

#### 1. **Audio Dashboard**
```javascript
// Connect to audio WebSocket
const audioWS = new WebSocket(`ws://localhost:8000/ws/audio?token=${token}`);

// Real-time audio graph
const audioGraph = new Chart(ctx, {
  type: 'line',
  data: { datasets: [{ data: [] }] }
});

audioWS.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  // Update waveform graph
  if (data.waveform) {
    audioGraph.data.datasets[0].data = data.waveform;
    audioGraph.update();
  }
  
  // Show cough alert
  if (data.event === 'prediction' && data.label === 'Cough') {
    showCoughAlert(data);
    addToHistory(data);
  }
};

// Fetch cough history
async function loadCoughHistory() {
  const response = await fetch('/api/cough/detections');
  const { items } = await response.json();
  
  items.forEach(item => {
    displayCoughEvent(item);
  });
}

// Play saved audio
function playCoughAudio(mediaUrl) {
  const audio = new Audio(`http://localhost:8000${mediaUrl}`);
  audio.play();
}
```

#### 2. **Video Dashboard**
```javascript
// Connect to video stream
const videoStream = new EventSource(
  'http://localhost:8000/video/stream?camera_index=0'
);

videoStream.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  // Update emotion display
  document.getElementById('emotion').textContent = data.mood;
  
  // Fall alert
  if (data.fall_detected) {
    showFallAlert(data.timestamp);
  }
};

// Display webcam feed
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    document.getElementById('video').srcObject = stream;
  });
```

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message here"
}
```

**Common Status Codes:**
- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized (invalid/missing token)
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable (e.g., fall detection disabled)

---

## Setup Instructions

### 1. Environment Setup
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file:
```env
DATABASE_URL=sqlite:///./caretaker.db
SECRET_KEY=your-secret-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=43200

# Optional: Email configuration
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_FROM=your-email@gmail.com
MAIL_PORT=587
MAIL_SERVER=smtp.gmail.com
```

### 3. Run Server
```bash
python main.py
# or
uvicorn main:app --reload
```

### 4. Access API
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Models Required

Place ML models in the following structure:
```
caretaker/
├── models/
│   ├── audio/
│   │   └── cough/
│   │       ├── models/
│   │       │   ├── yamnet_88.keras
│   │       │   └── yamnet_88_savedmodel/
│   │       └── preprocessor/
│   │           └── preprocessor_saved.pkl
│   └── video/
│       ├── emotion_recognition.py
│       └── fall_detection.py
└── backend/
    └── ...
```

---

## Testing Flow

1. **Signup**: POST `/signup` → Receive welcome email
2. **Login**: POST `/login` → Get access token
3. **Audio Stream**: Connect to WebSocket with token
4. **Video Stream**: Open SSE connection
5. **View History**: GET `/api/cough/detections`
6. **Play Audio**: Access `/media/cough/<filename>.wav`

---

## Notes

- **Audio Sampling**: 16kHz, mono, 16-bit PCM
- **Video FPS**: Configurable via `interval_ms` parameter
- **Token Expiry**: 30 days (configurable)
- **Fall Detection**: Requires `ultralytics` package (optional)
- **Email**: Fails silently if not configured
