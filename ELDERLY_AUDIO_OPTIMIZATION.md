# Elderly Voice Capture Optimization Guide

## Challenge

Capture accurate elderly voices for ML predictions while handling:
- ✅ Low-level elderly voices (weak, quiet speech)
- ✅ Variable background noise (low to high)
- ✅ Ensure audio is loud enough for model accuracy
- ✅ Preserve cough/breathing sounds

## Current System Analysis

### What's Already Working ✅

1. **Adaptive Noise Gate**
   - Adjusts to background noise automatically
   - `MIN_NOISE_THRESHOLD = 100` (very sensitive)
   - `MAX_NOISE_THRESHOLD = 800` (prevents over-adaptation)

2. **Noise Reduction**
   - Removes background hum, static, white noise
   - Preserves speech/cough frequencies

3. **Sensitive Settings**
   - `-45 dB` threshold (good for quiet voices)
   - `5ms` attack (catches sudden sounds)

### Current Limitations ⚠️

1. **No Volume Normalization** - Quiet voices stay quiet
2. **Fixed Noise Reduction** - May be too aggressive for some environments
3. **No Dynamic Range Compression** - Loud and quiet sounds have same treatment

## Recommended Solutions

### Solution 1: Audio Normalization (CRITICAL) ⭐

Add automatic volume normalization to ensure consistent loudness for ML model.

**Benefits:**
- ✅ Quiet elderly voices boosted to optimal level
- ✅ Loud sounds reduced to prevent clipping
- ✅ Consistent input for ML model
- ✅ Better prediction accuracy

**Implementation:**

```python
def normalize_audio(audio, target_level=-20.0):
    """
    Normalize audio to target dB level
    
    Args:
        audio: numpy array of audio samples
        target_level: target RMS level in dB (default -20 dB)
    
    Returns:
        Normalized audio array
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
    
    # Avoid division by zero
    if rms < 1e-6:
        return audio
    
    # Calculate current dB level
    current_db = 20 * np.log10(rms / 32768.0)
    
    # Calculate gain needed
    gain_db = target_level - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Apply gain with clipping prevention
    normalized = audio.astype(np.float32) * gain_linear
    normalized = np.clip(normalized, -32768, 32767)
    
    return normalized.astype(np.int16)
```

**Where to add:** After noise reduction, before saving

**Target levels:**
- `-20 dB` = Good for speech/coughs (recommended)
- `-15 dB` = Louder (for very quiet voices)
- `-25 dB` = Quieter (for loud environments)

### Solution 2: Dynamic Range Compression ⭐

Compress dynamic range to make quiet sounds louder and loud sounds quieter.

**Benefits:**
- ✅ Quiet whispers become audible
- ✅ Loud coughs don't clip
- ✅ More consistent audio levels
- ✅ Better for ML model

**Implementation:**

```python
def compress_audio(audio, threshold=-30, ratio=4.0, attack=0.005, release=0.1):
    """
    Apply dynamic range compression
    
    Args:
        audio: numpy array of audio samples
        threshold: compression threshold in dB
        ratio: compression ratio (4:1 means 4dB input = 1dB output above threshold)
        attack: attack time in seconds
        release: release time in seconds
    
    Returns:
        Compressed audio
    """
    from scipy import signal
    
    # Convert to float
    audio_float = audio.astype(np.float32) / 32768.0
    
    # Calculate envelope
    envelope = np.abs(audio_float)
    
    # Smooth envelope with attack/release
    smoothed = np.zeros_like(envelope)
    attack_coef = np.exp(-1.0 / (attack * 16000))
    release_coef = np.exp(-1.0 / (release * 16000))
    
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i-1]:
            smoothed[i] = attack_coef * smoothed[i-1] + (1 - attack_coef) * envelope[i]
        else:
            smoothed[i] = release_coef * smoothed[i-1] + (1 - release_coef) * envelope[i]
    
    # Calculate gain reduction
    threshold_linear = 10 ** (threshold / 20.0)
    gain = np.ones_like(smoothed)
    
    above_threshold = smoothed > threshold_linear
    gain[above_threshold] = (
        threshold_linear * (smoothed[above_threshold] / threshold_linear) ** (1.0 / ratio)
    ) / smoothed[above_threshold]
    
    # Apply compression
    compressed = audio_float * gain
    
    # Convert back to int16
    compressed = np.clip(compressed * 32768.0, -32768, 32767)
    return compressed.astype(np.int16)
```

### Solution 3: Adaptive Noise Reduction ⭐

Adjust noise reduction strength based on background noise level.

**Current:** Fixed `prop_decrease=0.5`

**Better:** Adaptive based on noise floor

```python
# Calculate noise level
noise_level = np.percentile(noise_floor_buffer, 50) if noise_floor_buffer else 100

# Adaptive noise reduction strength
if noise_level < 200:
    # Quiet environment - gentle noise reduction
    prop_decrease = 0.3
elif noise_level < 500:
    # Moderate noise - standard reduction
    prop_decrease = 0.5
else:
    # Noisy environment - aggressive reduction
    prop_decrease = 0.7

# Apply noise reduction
reduced_noise = nr.reduce_noise(
    y=data_np,
    sr=RATE,
    stationary=True,
    prop_decrease=prop_decrease
)
```

### Solution 4: Multi-Band Processing

Process different frequency bands separately (speech vs noise).

**Benefits:**
- ✅ Preserve speech frequencies (300-3400 Hz)
- ✅ Remove low-frequency rumble
- ✅ Remove high-frequency hiss
- ✅ Better speech clarity

```python
from scipy.signal import butter, sosfilt

def multi_band_process(audio, sr=16000):
    """
    Process audio in multiple frequency bands
    """
    # High-pass filter: Remove rumble below 100 Hz
    sos_hp = butter(4, 100, 'hp', fs=sr, output='sos')
    audio = sosfilt(sos_hp, audio)
    
    # Low-pass filter: Remove hiss above 8000 Hz
    sos_lp = butter(4, 8000, 'lp', fs=sr, output='sos')
    audio = sosfilt(sos_lp, audio)
    
    # Boost speech frequencies (300-3400 Hz)
    sos_bp = butter(2, [300, 3400], 'bp', fs=sr, output='sos')
    speech_band = sosfilt(sos_bp, audio)
    
    # Mix: 70% full spectrum + 30% speech boost
    enhanced = 0.7 * audio + 0.3 * speech_band
    
    return enhanced.astype(np.int16)
```

## Recommended Implementation Plan

### Phase 1: Quick Wins (Implement First) 🚀

1. **Add Audio Normalization**
   - Target: `-20 dB` RMS
   - Where: After noise reduction, before saving
   - Impact: **HIGH** - Ensures consistent volume for ML

2. **Adjust Current Settings**
   ```python
   # More sensitive for quiet voices
   NOISE_GATE_THRESHOLD = -50  # Was -45
   MIN_NOISE_THRESHOLD = 80    # Was 100
   
   # Gentler noise reduction
   prop_decrease = 0.4  # Was 0.5
   ```

### Phase 2: Advanced Processing 🎯

3. **Add Dynamic Range Compression**
   - Threshold: `-30 dB`
   - Ratio: `4:1`
   - Impact: **MEDIUM** - Evens out volume variations

4. **Adaptive Noise Reduction**
   - Adjust based on environment
   - Impact: **MEDIUM** - Better in varying conditions

### Phase 3: Fine-Tuning 🔧

5. **Multi-Band Processing**
   - Enhance speech frequencies
   - Impact: **LOW-MEDIUM** - Improves clarity

6. **Pre-emphasis Filter**
   - Boost high frequencies for speech
   - Impact: **LOW** - Subtle improvement

## Complete Processing Pipeline (Recommended)

```
1. Capture Audio (Microphone)
   ↓
2. High-Pass Filter (Remove rumble < 100 Hz)
   ↓
3. Adaptive Noise Reduction (Based on environment)
   ↓
4. Dynamic Range Compression (Even out levels)
   ↓
5. Audio Normalization (Target -20 dB)
   ↓
6. Multi-Band Enhancement (Boost speech frequencies)
   ↓
7. Save to File (Full volume, clear audio)
   ↓
8. Send to ML Model (Optimal input)
```

## Specific Code Changes

### Change 1: Add Normalization Function

**File:** `backend/routes/audio.py`

**Add after line 96:**

```python
def normalize_audio(audio, target_db=-20.0):
    """Normalize audio to target dB level for consistent ML input"""
    rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
    if rms < 1e-6:
        return audio
    current_db = 20 * np.log10(rms / 32768.0)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    normalized = audio.astype(np.float32) * gain_linear
    normalized = np.clip(normalized, -32768, 32767)
    return normalized.astype(np.int16)
```

### Change 2: Apply Normalization Before Saving

**File:** `backend/routes/audio.py`

**Find line ~295 (after loading audio):**

```python
y, sr = librosa.load(str(tmp_path), sr=None, mono=True)
if sr != router.target_sr:
    y = librosa.resample(y, orig_sr=sr, target_sr=router.target_sr)
    sr = router.target_sr

# ADD THIS: Normalize audio for consistent ML input
y_int16 = (y * 32768).astype(np.int16)
y_normalized = normalize_audio(y_int16, target_db=-20.0)
y = y_normalized.astype(np.float32) / 32768.0

# Continue with existing normalization
y = y / (np.max(np.abs(y)) + 1e-6)
```

### Change 3: Adaptive Noise Reduction

**File:** `backend/routes/audio.py`

**Find line ~215 (noise reduction section):**

```python
# Calculate adaptive noise reduction strength
noise_level = adaptive_threshold
if noise_level < 200:
    prop_decrease = 0.3  # Gentle for quiet environments
elif noise_level < 500:
    prop_decrease = 0.5  # Standard
else:
    prop_decrease = 0.7  # Aggressive for noisy environments

# Apply adaptive noise reduction
reduced_noise = nr.reduce_noise(
    y=data_np,
    sr=RATE,
    stationary=True,
    prop_decrease=prop_decrease
)
```

## Testing & Validation

### Test Scenarios

1. **Quiet Elderly Voice**
   - Whisper or speak quietly
   - Check: Audio should be boosted to -20 dB
   - Verify: Model can make accurate predictions

2. **Normal Voice with Background Noise**
   - Speak normally with TV/radio on
   - Check: Voice clear, background reduced
   - Verify: Consistent volume

3. **Loud Cough**
   - Cough loudly
   - Check: No clipping, normalized to -20 dB
   - Verify: Model detects accurately

4. **Variable Environment**
   - Test in quiet room, then noisy room
   - Check: Adaptive threshold adjusts
   - Verify: Consistent detection

### Metrics to Monitor

```python
# Log these for each saved audio
logging.info(f"Original RMS: {original_rms:.1f}")
logging.info(f"Normalized RMS: {normalized_rms:.1f}")
logging.info(f"Target dB: -20.0, Actual dB: {actual_db:.1f}")
logging.info(f"Noise floor: {adaptive_threshold:.1f}")
logging.info(f"Noise reduction strength: {prop_decrease:.2f}")
```

## Expected Results

### Before Optimization:
- Quiet voice: RMS = 200, dB = -60 ❌ (too quiet for model)
- Loud cough: RMS = 15000, dB = -10 ❌ (may clip)
- Inconsistent predictions ❌

### After Optimization:
- Quiet voice: RMS = 2500, dB = -20 ✅ (optimal for model)
- Loud cough: RMS = 2500, dB = -20 ✅ (normalized)
- Consistent, accurate predictions ✅

## Recommended Settings Summary

```python
# Noise Gate (More Sensitive)
NOISE_GATE_THRESHOLD = -50  # dB (was -45)
MIN_NOISE_THRESHOLD = 80    # RMS (was 100)
MAX_NOISE_THRESHOLD = 800   # RMS (unchanged)

# Normalization (NEW)
TARGET_DB_LEVEL = -20.0     # Target loudness

# Compression (NEW - Optional)
COMPRESSION_THRESHOLD = -30  # dB
COMPRESSION_RATIO = 4.0      # 4:1

# Noise Reduction (Adaptive)
PROP_DECREASE_MIN = 0.3     # Quiet environments
PROP_DECREASE_MED = 0.5     # Normal environments
PROP_DECREASE_MAX = 0.7     # Noisy environments
```

## Priority Implementation

**Must Have (Phase 1):**
1. ✅ Audio normalization to -20 dB
2. ✅ More sensitive noise gate (-50 dB)
3. ✅ Adaptive noise reduction

**Should Have (Phase 2):**
4. ✅ Dynamic range compression
5. ✅ High-pass filter (remove rumble)

**Nice to Have (Phase 3):**
6. ✅ Multi-band processing
7. ✅ Speech frequency enhancement

## Summary

The **#1 most important change** is **audio normalization** to ensure all audio (quiet or loud) is at a consistent level (-20 dB) for the ML model. This will dramatically improve prediction accuracy for elderly voices.

Combined with adaptive noise reduction and a more sensitive noise gate, you'll have a robust system that handles:
- ✅ Quiet elderly voices (boosted to optimal level)
- ✅ Variable background noise (adaptive processing)
- ✅ Loud sounds (compressed and normalized)
- ✅ Consistent ML model input (accurate predictions)
