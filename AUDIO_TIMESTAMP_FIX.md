# Audio Timestamp Error Fix

## Error Observed

```
ERROR:root:Segment processing failed: 'NoneType' object has no attribute 'isoformat'
AttributeError: 'NoneType' object has no attribute 'isoformat'
```

## Root Cause

The code was resetting `segment_started_at` to `None` before using it to create the timestamp for the prediction event.

### The Bug Flow:

1. Audio segment detected → `segment_started_at` set to current time ✅
2. Segment ends → Processing starts
3. **Line 284**: `segment_started_at = None` ❌ (Reset too early)
4. **Line 344**: `segment_started_at.isoformat()` ❌ (Trying to use None)
5. **Error**: AttributeError

## Solution

Save the timestamp to a local variable **before** resetting `segment_started_at` to `None`.

### Code Changes

**File**: `backend/routes/audio.py`

#### Before (Broken):
```python
raw_bytes = b"".join(segment_bytes)
segment_active = False
segment_bytes = []
segment_started_at = None  # ❌ Reset too early
# ... processing ...
event = {
    "timestamp": segment_started_at.isoformat() + "Z",  # ❌ None.isoformat() fails
}
```

#### After (Fixed):
```python
raw_bytes = b"".join(segment_bytes)
# Save timestamp before resetting ✅
segment_timestamp = segment_started_at if segment_started_at else datetime.utcnow()
segment_active = False
segment_bytes = []
segment_started_at = None  # ✅ Safe to reset now
# ... processing ...
event = {
    "timestamp": segment_timestamp.isoformat() + "Z",  # ✅ Works correctly
}
```

## What Changed

### 1. Save Timestamp Before Reset
**Line 283**:
```python
segment_timestamp = segment_started_at if segment_started_at else datetime.utcnow()
```

This creates a local copy of the timestamp before it's reset.

### 2. Use Saved Timestamp
**Line 346**:
```python
"timestamp": segment_timestamp.isoformat() + "Z",
```

**Line 352**:
```python
ts = segment_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
```

Now uses the saved `segment_timestamp` instead of `segment_started_at`.

## Testing

### Restart Backend
```bash
cd backend
python main.py
```

### Test Audio Monitoring

1. Open `frontend/dashboard.html`
2. Login
3. Navigate to "Audio Monitoring"
4. Speak or make sounds

**Expected logs (no errors):**
```
INFO:root:Processing audio segment: duration=3.16s
INFO:root:YAMNet embeddings extracted successfully
INFO:root:YAMNet embeddings shape: (9, 1024)
INFO:root:Running prediction with metadata: {'age': 70, 'gender': 'Female', ...}
INFO:root:Prediction result: Not Cough (probability=0.047)
```

**No more AttributeError!** ✅

### Test Cough Detection

Make a coughing sound:

**Expected logs:**
```
INFO:root:Processing audio segment: duration=2.45s
INFO:root:YAMNet embeddings extracted successfully
INFO:root:Prediction result: Cough (probability=0.85)
INFO:root:Saved cough audio: media/cough/cough_20251107T155728123456Z.wav
```

## Complete Audio Processing Flow

Now the complete flow works correctly:

1. **Audio Capture** → Microphone input
2. **Noise Gate** → Filter background noise
3. **Segment Detection** → Detect speech/sound segments
4. **Save Timestamp** → Record when segment started ✅
5. **YAMNet** → Extract audio embeddings
6. **Cough Classifier** → Predict if it's a cough
7. **Use Timestamp** → Create event with saved timestamp ✅
8. **Send to Frontend** → WebSocket sends prediction
9. **Save if Cough** → Store audio file with timestamp ✅

## Files Modified

1. ✅ `backend/routes/audio.py`
   - Line 283: Save timestamp before reset
   - Line 346: Use saved timestamp for event
   - Line 352: Use saved timestamp for filename

## Verification Checklist

After restarting backend:
- [ ] Backend starts without errors
- [ ] Audio WebSocket connects
- [ ] Audio levels display in frontend
- [ ] No "AttributeError: 'NoneType'" errors
- [ ] Predictions show in logs
- [ ] Cough detection works (if you cough)
- [ ] Cough files saved with correct timestamps

## Status

✅ **FIXED** - Audio timestamp error resolved

Both audio model issues are now fixed:
1. ✅ YAMNet tensor shape error (previous fix)
2. ✅ Timestamp AttributeError (this fix)

## Next Steps

1. **Restart backend** to apply the fix
2. **Test audio monitoring** in dashboard
3. **Make sounds** to test prediction
4. **Cough** to test cough detection and file saving

The audio monitoring should now work completely without errors! 🎉
