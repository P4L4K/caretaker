import numpy as np
import librosa
from config import AUDIO_TARGET_SR, AUDIO_RATE

def preprocess_audio_chunk(np_data, rate=None):
    """Normalize & resample the audio chunk."""
    if rate is None:
        rate = AUDIO_RATE

    normalized = np_data.astype(np.float32) / (np.max(np.abs(np_data)) + 1e-6)
    
    if rate != AUDIO_TARGET_SR:
        normalized = librosa.resample(normalized, orig_sr=rate, target_sr=AUDIO_TARGET_SR)
    
    return normalized
