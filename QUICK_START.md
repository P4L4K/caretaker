# 🚀 CareTaker AI - Quick Start Guide

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd backend
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Create Environment File
Create `backend/.env`:
```env
DATABASE_URL=sqlite:///./caretaker.db
SECRET_KEY=change-this-secret-key-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=43200
```

### Step 3: Start Server
```bash
python main.py
```

✅ Server running at: **http://localhost:8000**

---

## 🧪 Test the API

### Option 1: Run Test Script
```bash
python test_api.py
```

### Option 2: Use API Docs
Open browser: **http://localhost:8000/docs**

### Option 3: Manual Test
```bash
# 1. Signup
curl -X POST http://localhost:8000/signup \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john",
    "email": "john@example.com",
    "password": "Pass123",
    "care_recipients": [{
      "name": "Jane",
      "age": 75,
      "gender": "Female",
      "respiratory_condition_status": false
    }]
  }'

# 2. Login
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "john", "password": "Pass123"}'

# 3. Get token from response and test audio endpoint
curl http://localhost:8000/api/cough/detections
```

---

## 🎯 Connect Frontend

### Audio Monitoring (WebSocket)
```javascript
const token = "YOUR_ACCESS_TOKEN";
const ws = new WebSocket(`ws://localhost:8000/ws/audio?token=${token}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.waveform) {
    // Update audio graph
    console.log('Audio:', data.rms, data.db);
  }
  
  if (data.event === 'prediction' && data.label === 'Cough') {
    console.log('🔔 Cough detected!', data.timestamp);
    // Play audio: data.media_url
  }
};
```

### Video Monitoring (SSE)
```javascript
const stream = new EventSource('http://localhost:8000/video/stream');

stream.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Emotion:', data.mood);
  
  if (data.fall_detected) {
    alert('⚠️ FALL DETECTED!');
  }
};
```

---

## 📊 Expected Behavior

### Server Startup Logs
```
INFO:root:PyAudio instance created.
INFO:root:Loading models from: E:\...\models\audio\cough\models
INFO:root:Cough classifier loaded successfully!
INFO:root:YAMNet model loaded successfully!
INFO:root:Preprocessor loaded successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Audio Stream Logs (Every 5 seconds)
```
INFO:root:Audio stream started for user: john
INFO:root:RMS: 2500.0, dB: -22.5, Gate: 1.00
INFO:root:Forcing segment processing due to max duration: 5.00s
INFO:root:Processing audio segment: duration=5.00s
INFO:root:YAMNet embeddings shape: (X, 1024)
INFO:root:Running prediction with metadata: {'age': 75, ...}
INFO:root:Prediction result: Cough (probability=0.876)
```

---

## 🎨 Dashboard Features

### Audio Dashboard
- ✅ Real-time waveform graph
- ✅ Cough detection alerts
- ✅ Timestamp display
- ✅ Saved audio playback
- ✅ Search/filter history

### Video Dashboard
- ✅ Live webcam feed
- ✅ Emotion detection
- ✅ Fall alerts
- ✅ Event timestamps

---

## 📚 Full Documentation

- **Complete Guide**: `FIXES_AND_IMPROVEMENTS.md`
- **API Reference**: `backend/API_DOCUMENTATION.md`
- **Interactive Docs**: http://localhost:8000/docs

---

## 🆘 Troubleshooting

### No predictions?
1. Check models loaded (see startup logs)
2. Verify audio input device
3. Confirm WebSocket connected
4. Check token validity

### Fall detection not working?
```bash
pip install ultralytics
```
(Optional feature)

### Email not sending?
Configure SMTP in `.env` or ignore (optional feature)

---

## ✅ Success Checklist

- [ ] Server starts without errors
- [ ] Can signup new user
- [ ] Can login and get token
- [ ] WebSocket connects successfully
- [ ] Audio waveform updates in real-time
- [ ] Cough predictions appear every ~5 seconds
- [ ] Video stream shows emotion
- [ ] Can retrieve cough history
- [ ] Can play saved audio files

---

**🎉 You're all set! Start building your dashboard!**
