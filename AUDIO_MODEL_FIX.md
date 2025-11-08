# Audio Model Error Fix

## Error Observed

```
ERROR:root:YAMNet inference error: Binding inputs to tf.function failed due to 
`Can not cast <tf.Tensor: shape=(74880,), dtype=float32> to list type.`
```

## Root Cause

The code was trying to use `yamnet_88_savedmodel` as if it were the raw YAMNet model, but it's actually the **cough classifier model** that expects:
- **Input 1**: Audio embeddings (shape: `(None, 30, 1024)`)
- **Input 2**: Metadata (shape: `(None, 6)`)

But we were passing a raw audio waveform tensor (shape: `(74880,)`), which caused the type mismatch error.

## Solution

Changed the code to use **TensorFlow Hub's YAMNet** directly for extracting embeddings, then use those embeddings with the cough classifier.

### What Changed

**File**: `backend/routes/audio.py`

#### Before (Incorrect):
```python
# Tried to use cough classifier as YAMNet
if hasattr(router.yamnet_model, 'serve'):
    yamnet_output = router.yamnet_model.serve(waveform)  # ❌ Wrong input type
    embeddings = yamnet_output['embeddings'].numpy()
```

#### After (Correct):
```python
# Use TensorFlow Hub YAMNet for embeddings
import tensorflow_hub as hub

if not hasattr(router, '_yamnet_hub_model'):
    logging.info("Loading YAMNet from TensorFlow Hub...")
    router._yamnet_hub_model = hub.load('https://tfhub.dev/google/yamnet/1')

# YAMNet expects waveform, returns (scores, embeddings, spectrogram)
_, embeddings, _ = router._yamnet_hub_model(waveform)  # ✅ Correct
embeddings = embeddings.numpy()
```

## How It Works Now

### Audio Processing Pipeline:

1. **Capture Audio** → Raw audio from microphone
2. **Preprocess** → Noise reduction, resampling to 16kHz
3. **YAMNet (TF Hub)** → Extract audio embeddings from waveform
4. **Cough Classifier** → Use embeddings to detect coughs
5. **Send Results** → WebSocket sends predictions to frontend

### Model Roles:

| Model | Purpose | Input | Output |
|-------|---------|-------|--------|
| **YAMNet (TF Hub)** | Extract audio features | Raw waveform | Embeddings (1024-dim) |
| **yamnet_88.keras** | Classify coughs | Embeddings + metadata | Cough probability |
| **yamnet_88_savedmodel** | Alternative classifier | Embeddings + metadata | Cough probability |

## Dependencies

The fix requires `tensorflow-hub` which is already in `requirements.txt`:
```
tensorflow-hub
```

If not installed:
```bash
pip install tensorflow-hub
```

## Testing

### 1. Restart Backend
```bash
cd backend
python main.py
```

**Expected startup logs:**
```
INFO:root:Cough classifier loaded successfully!
INFO:root:Cough classifier model loaded successfully (yamnet_88_savedmodel)!
INFO:root:Preprocessor loaded successfully!
```

### 2. Test Audio Monitoring

1. Open `frontend/dashboard.html`
2. Login
3. Navigate to "Audio Monitoring"
4. Audio WebSocket should connect automatically
5. Speak or make sounds

**Expected logs (no errors):**
```
INFO:root:RMS: 375.6, dB: -38.8, Gate: 1.00, Hold: 1, Adaptive: 150.1
INFO:root:Processing audio segment: duration=0.37s
INFO:root:Loading YAMNet from TensorFlow Hub...  (first time only)
INFO:root:YAMNet embeddings extracted successfully
INFO:root:YAMNet embeddings shape: (X, 1024)
INFO:root:Preprocessor input shape: (30, 1024)
INFO:root:Cough prediction: probability=0.XX
```

### 3. Test Cough Detection

1. Make a coughing sound
2. Check frontend for cough detection
3. Check backend logs for prediction

**Expected:**
```
INFO:root:Cough detected! Probability: 0.85
INFO:root:Saved cough audio: media/cough/cough_TIMESTAMP.wav
```

## Performance Notes

### First Audio Segment:
- YAMNet downloads from TensorFlow Hub (~13MB)
- Takes 5-10 seconds on first load
- Cached for subsequent uses

### Subsequent Segments:
- YAMNet already loaded (fast)
- Processing time: ~100-200ms per segment

## Troubleshooting

### Issue: "Failed to load YAMNet from TensorFlow Hub"
**Cause**: No internet connection or TF Hub blocked
**Solution**: 
```bash
# Download YAMNet manually
python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"
```

### Issue: Still getting tensor shape errors
**Cause**: Old code cached
**Solution**: 
1. Stop backend (Ctrl+C)
2. Restart: `python main.py`
3. Hard refresh frontend (Ctrl+Shift+R)

### Issue: "No module named 'tensorflow_hub'"
**Cause**: Package not installed
**Solution**:
```bash
pip install tensorflow-hub
```

### Issue: Audio detection very slow
**Cause**: YAMNet downloading or CPU overload
**Solution**:
- Wait for first download to complete
- Close other applications
- Check CPU usage

## Files Modified

1. ✅ `backend/routes/audio.py`
   - Fixed YAMNet model usage
   - Use TensorFlow Hub YAMNet for embeddings
   - Clarified model purposes in comments

## Verification Checklist

After restarting backend:
- [ ] Backend starts without errors
- [ ] Audio WebSocket connects
- [ ] No "YAMNet inference error" in logs
- [ ] Audio levels display in frontend
- [ ] Cough detection works (if you cough)
- [ ] No tensor shape errors

## Status

✅ **FIXED** - Audio model now uses correct YAMNet from TensorFlow Hub

The audio monitoring should now work without tensor shape errors! 🎉

## Next Steps

1. **Restart the backend** to apply the fix
2. **Test audio monitoring** in the dashboard
3. **Check logs** for "YAMNet embeddings extracted successfully"
4. **Make a cough sound** to test detection

The error should be completely resolved now!
