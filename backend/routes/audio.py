 
import asyncio
import json
import logging
import numpy as np
import pyaudio
import noisereduce as nr
import wave
import os
from datetime import datetime
from pathlib import Path
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import load_model as keras_load_model
import joblib
import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Depends
from scipy.io.wavfile import write as write_wav
from sqlalchemy.orm import Session
from tables.users import CareTaker, CareRecipient
from repository.users import UsersRepo
from repository.users import JWTRepo
from repository.token_blocklist import TokenBlocklistRepo
from repository import cough_detections as cough_repo
from config import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
CHUNK = 4096
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
WAVE_OUTPUT_FILENAME = "output.wav"

# Noise gate parameters - HIGHLY OPTIMIZED FOR ELDERLY CARE
# Designed for VERY QUIET elderly voices and weak coughs
# Target sounds: weak coughs, snores, groans, breathing irregularities, falls, calls for help
NOISE_GATE_THRESHOLD = -55  # dB threshold (VERY sensitive for weak elderly voices)
SILENCE_THRESHOLD_RMS = 300  # MUCH lower for very quiet elderly sounds
# CRITICAL: Fast attack for transient detection (coughs, falls)
ATTACK_TIME = 0.003  # 3ms - Even faster to catch weak transients
# Longer release to capture full weak coughs
RELEASE_TIME = 0.25  # 250ms - Longer to capture complete weak coughs
# Extended hold time for weak, labored breathing patterns
HOLD_TIME = 0.15  # 150ms - Much longer hold for weak elderly speech

router = APIRouter()
p = pyaudio.PyAudio()

@router.on_event("startup")
async def startup_event():
    router.pyaudio_instance = pyaudio.PyAudio()
    logging.info("PyAudio instance created.")
    # Get absolute path to models directory (parent of backend)
    backend_dir = Path(__file__).parent.parent
    models_dir = backend_dir.parent / "models" / "audio" / "cough" / "models"
    preproc_dir = backend_dir.parent / "models" / "audio" / "cough" / "preprocessor"
    logging.info(f"Loading models from: {models_dir}")
    router.cough_classifier = None
    try:
        router.cough_classifier = keras_load_model(str(models_dir / "yamnet_88.keras"))
        logging.info("Cough classifier loaded successfully!")
    except Exception as e1:
        try:
            router.cough_classifier = tf.keras.models.load_model(str(models_dir / "yamnet_88.keras"), compile=False)
            logging.info("Cough classifier loaded successfully (TF method)!")
        except Exception as e2:
            logging.error(f"Failed to load cough classifier: {e1} | {e2}")
            router.cough_classifier = None
    # Note: yamnet_88_savedmodel is actually the cough classifier, not raw YAMNet
    # We'll load actual YAMNet from TensorFlow Hub when needed
    try:
        router.yamnet_model = tf.saved_model.load(str(models_dir / "yamnet_88_savedmodel"))
        logging.info("Cough classifier model loaded successfully (yamnet_88_savedmodel)!")
    except Exception as e:
        router.yamnet_model = None
        logging.error(f"Failed to load cough classifier model: {e}")
    try:
        router.preprocessor = joblib.load(str(preproc_dir / "preprocessor_saved.pkl"))
        logging.info("Preprocessor loaded successfully!")
    except Exception as e:
        router.preprocessor = None
        logging.error(f"Failed to load preprocessor: {e}")
    router.media_dir = Path("media/cough")
    router.media_dir.mkdir(parents=True, exist_ok=True)
    router.target_sr = 16000
    router.target_duration = 4.68
    router.max_segments = 30
    router.threshold = 0.5

def calculate_db(rms):
    """Convert RMS to decibels."""
    if rms <= 0:
        return -100
    return 20 * np.log10(rms / 32768.0)

def normalize_audio(audio, target_db=-20.0):
    """
    Normalize audio to target dB level for consistent ML input.
    Critical for elderly voices which can be very quiet.
    
    Args:
        audio: numpy array of int16 audio samples
        target_db: target RMS level in dB (default -20 dB, optimal for speech)
    
    Returns:
        Normalized audio array (int16)
    """
    rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
    if rms < 1e-6:
        return audio
    
    current_db = 20 * np.log10(rms / 32768.0)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    
    # Apply gain with clipping prevention
    normalized = audio.astype(np.float32) * gain_linear
    normalized = np.clip(normalized, -32768, 32767)
    
    logging.info(f"Audio normalization: {current_db:.1f} dB → {target_db:.1f} dB (gain: {gain_db:+.1f} dB)")
    
    return normalized.astype(np.int16)

def enhance_audio_for_elderly(y, sr=16000):
    """
    Enhance audio specifically for elderly voice detection.
    Improves weak cough detection while preserving natural characteristics.
    
    Args:
        y: Audio signal (float32, normalized)
        sr: Sample rate
    
    Returns:
        Enhanced audio signal (float32)
    """
    try:
        enhanced = y.copy()
        
        # 1. High-pass filter to remove low-frequency noise
        # Elderly coughs are typically 200-2000 Hz, remove below 80 Hz
        from scipy import signal
        nyquist = sr // 2
        low_cutoff = 80 / nyquist
        high_cutoff = 4000 / nyquist  # Remove high-frequency noise above 4kHz
        
        # Band-pass filter (80Hz - 4kHz) - optimal for cough detection
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        enhanced = signal.filtfilt(b, a, enhanced)
        
        # 2. Spectral enhancement - boost cough frequency ranges
        # Coughs have strong energy in 200-800 Hz and 1000-2000 Hz
        stft = librosa.stft(enhanced, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Create frequency mask for enhancement
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Boost cough-dominant frequencies
        freq_mask = np.ones_like(freqs)
        # Primary cough range: 200-800 Hz
        freq_mask[(freqs >= 200) & (freqs <= 800)] = 1.5
        # Secondary cough range: 1000-2000 Hz  
        freq_mask[(freqs >= 1000) & (freqs <= 2000)] = 1.3
        # Reduce noise frequencies
        freq_mask[freqs < 100] = 0.5  # Reduce very low frequencies
        freq_mask[freqs > 3000] = 0.8  # Reduce high frequencies
        
        # Apply frequency weighting
        enhanced_magnitude = magnitude * freq_mask[:, np.newaxis]
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft, hop_length=512)
        
        # 3. Dynamic range compression for weak sounds
        # Helps bring out weak cough characteristics
        threshold = 0.6
        ratio = 4.0
        
        # Apply soft compression
        abs_enhanced = np.abs(enhanced)
        mask = abs_enhanced > threshold
        compressed = np.where(mask, 
                             threshold + (abs_enhanced - threshold) / ratio,
                             abs_enhanced)
        enhanced = np.sign(enhanced) * compressed
        
        # 4. Gentle noise reduction specifically for speech
        # Use spectral subtraction with conservative parameters
        stft = librosa.stft(enhanced, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        
        # Estimate noise from quiet portions
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True) * 0.3
        
        # Spectral subtraction with over-subtraction factor
        alpha = 0.1  # Conservative over-subtraction
        beta = 0.01  # Spectral floor
        
        enhanced_magnitude = magnitude - alpha * noise_floor
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * np.angle(stft))
        enhanced = librosa.istft(enhanced_stft, hop_length=512)
        
        # 5. Final normalization to prevent clipping
        max_val = np.max(np.abs(enhanced))
        if max_val > 0:
            enhanced = enhanced / max_val * 0.95
        
        # 6. Ensure same length as input
        if len(enhanced) != len(y):
            if len(enhanced) > len(y):
                enhanced = enhanced[:len(y)]
            else:
                enhanced = np.pad(enhanced, (0, len(y) - len(enhanced)))
        
        logging.info(f"Audio enhancement applied: {len(y)} samples → {len(enhanced)} samples")
        return enhanced
        
    except Exception as e:
        logging.warning(f"Audio enhancement failed, using original: {e}")
        return y

def apply_noise_gate_with_hold(audio_data, rms, gate_state, hold_counter, 
                                attack_samples, release_samples, hold_samples, adaptive_threshold):
    """
    Apply noise gate with attack, hold, and release envelope.
    Returns: (gated_audio, new_gate_state, new_hold_counter)
    """
    db_level = calculate_db(rms)
    
    # Determine if gate should be open
    # Replace with adaptive threshold
    gate_open = db_level > NOISE_GATE_THRESHOLD and rms > adaptive_threshold
    
    # Prevent division by zero
    attack_samples = max(1, attack_samples)
    release_samples = max(1, release_samples)
    
    # Smooth gate transitions with hold
    if gate_open:
        # Attack phase - gate opening
        gate_state = min(1.0, gate_state + 1.0 / attack_samples)
        hold_counter = hold_samples  # Reset hold counter
    elif hold_counter > 0:
        # Hold phase - keep gate open
        gate_state = 1.0
        hold_counter -= 1
    elif gate_state > 0.0:
        # Release phase - gate closing
        gate_state = max(0.0, gate_state - 1.0 / release_samples)
    
    # Apply gate with smooth envelope
    gated_audio = audio_data * gate_state
    
    return gated_audio.astype(np.int16), gate_state, hold_counter

async def audio_stream(websocket: WebSocket, token: dict, db: Session):
    """Capture, process, and stream audio data."""
    p = router.pyaudio_instance

    # Extract username from token and get user data
    username = token.get("sub")
    caretaker = UsersRepo.find_by_username(db, CareTaker, username) if username else None
    
    # Get first care recipient for metadata (you may want to make this configurable)
    recipient = None
    if caretaker and caretaker.care_recipients:
        recipient = caretaker.care_recipients[0]
    
    # Create metadata dictionary for model prediction
    meta_dict = {
        "age": recipient.age if recipient else 30,
        "gender": recipient.gender.value if recipient and recipient.gender else "Male",
        "respiratory_condition": recipient.respiratory_condition_status if recipient else False
    }
    
    logging.info(f"Audio stream started for user: {username}, metadata: {meta_dict}")

    device_info = p.get_default_input_device_info()
    logging.info(f"Using device: {device_info.get('name')}")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    await websocket.accept()
    logging.info("WebSocket connection accepted.")
    
    audio_buffer = []
    gate_state = 0.0  # Gate envelope state (0.0 = closed, 1.0 = open)
    hold_counter = 0  # NEW: Track hold time
    segment_active = False
    segment_bytes = []
    segment_started_at = None
    max_segment_duration = 5.0  # Maximum segment duration in seconds before forcing processing
    
    # Calculate timing in samples
    attack_samples = max(1, int(ATTACK_TIME * RATE / CHUNK))
    release_samples = max(1, int(RELEASE_TIME * RATE / CHUNK))
    hold_samples = max(1, int(HOLD_TIME * RATE / CHUNK))  # NEW
    
    logging.info(f"Gate timing - Attack: {attack_samples}, Release: {release_samples}, Hold: {hold_samples}")
    logging.info(f"Attack time: {ATTACK_TIME*1000:.1f}ms, Release: {RELEASE_TIME*1000:.1f}ms, Hold: {HOLD_TIME*1000:.1f}ms")
    logging.info(f"Noise gate threshold: {NOISE_GATE_THRESHOLD} dB, RMS threshold: {SILENCE_THRESHOLD_RMS}")
    
    # Rolling buffer for adaptive noise floor estimation
    noise_floor_buffer = []
    noise_floor_size = 100  # Larger buffer for stable noise floor in care environment
    MIN_NOISE_THRESHOLD = 50   # VERY low baseline for extremely quiet elderly voices
    MAX_NOISE_THRESHOLD = 500  # Lower cap to stay sensitive to weak sounds
    
    try:
        while True:
            # Check if WebSocket is still connected before processing
            try:
                # Quick ping to detect disconnection early
                if websocket.client_state.name != "CONNECTED":
                    logging.info("WebSocket disconnected, stopping audio stream")
                    break
            except Exception:
                logging.info("WebSocket connection check failed, stopping audio stream")
                break
            
            data = stream.read(CHUNK, exception_on_overflow=False)
            data_np = np.frombuffer(data, dtype=np.int16)
            
            # Calculate initial RMS
            rms_raw = np.sqrt(np.mean(np.square(data_np.astype(np.float32))))
            
            # Adaptive noise floor estimation - HIGHLY SENSITIVE FOR ELDERLY
            # Only update noise floor with VERY LOW values (ambient noise, not patient sounds)
            if rms_raw < SILENCE_THRESHOLD_RMS * 0.5:  # Only track very quiet ambient noise
                if len(noise_floor_buffer) >= noise_floor_size:
                    noise_floor_buffer.pop(0)
                noise_floor_buffer.append(rms_raw)
            
            # Calculate adaptive threshold - MORE SENSITIVE
            if len(noise_floor_buffer) > 20:
                adaptive_threshold = np.clip(
                    np.percentile(noise_floor_buffer, 75) * 1.2,  # Lower percentile and multiplier
                    MIN_NOISE_THRESHOLD,
                    MAX_NOISE_THRESHOLD
                )
            else:
                adaptive_threshold = MIN_NOISE_THRESHOLD
            
            # Always apply noise reduction for consistent processing
            # VERY GENTLE noise reduction to preserve weak elderly speech
            reduced_noise = nr.reduce_noise(
                y=data_np, 
                sr=RATE, 
                stationary=True,
                prop_decrease=0.3  # Much less aggressive to preserve very weak sounds
            )
            
            # Calculate RMS after noise reduction
            rms = np.sqrt(np.mean(np.square(reduced_noise.astype(np.float32))))
            
            # Apply noise gate with hold
            gated_audio, gate_state, hold_counter = apply_noise_gate_with_hold(
                reduced_noise, rms, gate_state, hold_counter,
                attack_samples, release_samples, hold_samples, adaptive_threshold
            )
            
            # Log audio levels for debugging (comment out after tuning)
            db_level = calculate_db(rms)
            logging.info(f"RMS: {rms:.1f}, dB: {db_level:.1f}, Gate: {gate_state:.2f}, Hold: {hold_counter}, Adaptive: {adaptive_threshold:.1f}")
            
            # Check if segment has exceeded maximum duration
            should_process_segment = False
            if segment_active and segment_started_at:
                elapsed = (datetime.utcnow() - segment_started_at).total_seconds()
                if elapsed >= max_segment_duration:
                    should_process_segment = True
                    logging.info(f"Forcing segment processing due to max duration: {elapsed:.2f}s")
            
            # Only save and send if gate is significantly open
            if gate_state > 0.1:
                audio_buffer.append(gated_audio.tobytes())
                # Save ORIGINAL RAW audio (before noise reduction) for playback
                # This ensures saved files are audible and not silent
                segment_bytes.append(data_np.tobytes())
                if not segment_active:
                    segment_active = True
                    segment_started_at = datetime.utcnow()
                
                # VISUALIZATION THRESHOLD: Only show waveform if gate is substantially open
                # This prevents minor noise from cluttering the display while still detecting coughs
                display_threshold = 0.3  # Show waveform only when gate is 30% or more open
                
                if gate_state >= display_threshold:
                    # Show actual waveform for significant sounds
                    display_waveform = gated_audio.tolist()
                else:
                    # Show minimal waveform for weak sounds (smooth visualization)
                    display_waveform = (gated_audio * 0.2).astype(np.int16).tolist()
                
                # Send waveform data to the frontend
                try:
                    await websocket.send_text(json.dumps({
                        "waveform": display_waveform,
                        "rms": float(rms),
                        "db": float(db_level),
                        "gate_open": gate_state > 0.5,
                        "gate_level": float(gate_state),
                        "hold_active": hold_counter > 0
                    }))
                except Exception as e:
                    logging.info(f"Failed to send waveform data, client disconnected: {e}")
                    break
                
                # Process segment if max duration reached
                if should_process_segment:
                    # Don't wait for gate to close, process immediately
                    pass  # Will be processed below
            else:
                # Send silence indicator
                try:
                    await websocket.send_text(json.dumps({
                        "waveform": [0] * len(gated_audio),
                        "rms": 0.0,
                        "db": -100.0,
                        "gate_open": False,
                        "gate_level": 0.0,
                        "hold_active": False
                    }))
                except Exception as e:
                    logging.info(f"Failed to send silence data, client disconnected: {e}")
                    break
            
            # Process segment when gate closes OR max duration reached
            if (gate_state <= 0.1 or should_process_segment) and segment_active and segment_bytes:
                    try:
                        raw_bytes = b"".join(segment_bytes)
                        # Save timestamp before resetting
                        segment_timestamp = segment_started_at if segment_started_at else datetime.utcnow()
                        segment_active = False
                        segment_bytes = []
                        segment_started_at = None  # Reset timer
                        duration_sec = len(raw_bytes) / (2 * RATE)
                        if duration_sec >= 0.2:  # Lower minimum duration for short weak coughs
                            tmp_path = router.media_dir / "_tmp_segment.wav"
                            data_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
                            sf.write(str(tmp_path), data_i16.astype(np.int16), RATE, subtype='PCM_16')

                            y, sr = librosa.load(str(tmp_path), sr=None, mono=True)
                            if sr != router.target_sr:
                                y = librosa.resample(y, orig_sr=sr, target_sr=router.target_sr)
                                sr = router.target_sr
                            y = y / (np.max(np.abs(y)) + 1e-6)
                            target_length = int(router.target_duration * sr)
                            if len(y) > target_length:
                                y = y[:target_length]
                            elif len(y) < target_length:
                                y = np.pad(y, (0, target_length - len(y)))

                            # ENHANCE AUDIO FOR ELDERLY DETECTION
                            # Apply specialized enhancement before ML processing
                            y_enhanced = enhance_audio_for_elderly(y, sr=router.target_sr)
                            
                            if router.yamnet_model is None or router.cough_classifier is None or router.preprocessor is None:
                                logging.warning(f"Skipping prediction: yamnet={router.yamnet_model is not None}, classifier={router.cough_classifier is not None}, preprocessor={router.preprocessor is not None}")
                                continue

                            logging.info(f"Processing audio segment: duration={duration_sec:.2f}s (enhanced)")
                            waveform = tf.convert_to_tensor(y_enhanced, dtype=tf.float32)
                            
                            # Call YAMNet to get embeddings
                            try:
                                # Use TensorFlow Hub YAMNet directly for embeddings
                                import tensorflow_hub as hub
                                
                                # Load YAMNet from TF Hub on-the-fly (cached after first load)
                                if not hasattr(router, '_yamnet_hub_model'):
                                    logging.info("Loading YAMNet from TensorFlow Hub...")
                                    router._yamnet_hub_model = hub.load('https://tfhub.dev/google/yamnet/1')
                                
                                # YAMNet expects waveform, returns (scores, embeddings, spectrogram)
                                _, embeddings, _ = router._yamnet_hub_model(waveform)
                                embeddings = embeddings.numpy()
                                
                                logging.info(f"YAMNet embeddings extracted successfully")
                            except Exception as e:
                                logging.error(f"YAMNet inference error: {e}")
                                raise
                            
                            logging.info(f"YAMNet embeddings shape: {embeddings.shape}")
                            if embeddings.shape[0] < router.max_segments:
                                embeddings = np.pad(embeddings, ((0, router.max_segments - embeddings.shape[0]), (0, 0)), mode='constant')
                            else:
                                embeddings = embeddings[:router.max_segments]

                            meta_df = pd.DataFrame([meta_dict])
                            meta_input = router.preprocessor.transform(meta_df).astype('float32')
                            logging.info(f"Running prediction with metadata: {meta_dict}")
                            pred = router.cough_classifier.predict([embeddings[np.newaxis, :, :], meta_input], verbose=0)
                            prob = float(pred[0, 0])
                            label = "Cough" if prob >= router.threshold else "Not Cough"
                            logging.info(f"Prediction result: {label} (probability={prob:.3f})")

                            event = {
                                "event": "prediction",
                                "timestamp": segment_timestamp.isoformat() + "Z",
                                "probability": prob,
                                "label": label
                            }

                            if label == "Cough":
                                ts = segment_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
                                out_path = router.media_dir / f"cough_{ts}.wav"
                                
                                # Normalize audio to -20 dB for consistent volume (critical for elderly voices)
                                y_int16 = (y * 32768).astype(np.int16)
                                y_normalized = normalize_audio(y_int16, target_db=-20.0)
                                
                                # Save normalized audio
                                sf.write(str(out_path), y_normalized, sr, subtype='PCM_16')
                                event["media_url"] = f"/media/cough/{out_path.name}"
                                sidecar = {"timestamp": event["timestamp"], "probability": prob, "label": label, "media_url": event["media_url"], "username": username, "caretaker_id": (caretaker.id if caretaker else None), "recipient_id": (recipient.id if recipient else None), "age": meta_dict["age"], "gender": meta_dict["gender"], "respiratory_condition": meta_dict["respiratory_condition"]}
                                with open(str(out_path.with_suffix('.json')), 'w', encoding='utf-8') as f:
                                    json.dump(sidecar, f)
                                
                                # Save to database
                                try:
                                    detection_data = {
                                        "timestamp": segment_timestamp,
                                        "probability": prob,
                                        "label": label,
                                        "media_url": event["media_url"],
                                        "username": username,
                                        "caretaker_id": caretaker.id if caretaker else None,
                                        "recipient_id": recipient.id if recipient else None,
                                        "age": meta_dict["age"],
                                        "gender": meta_dict["gender"],
                                        "respiratory_condition": meta_dict["respiratory_condition"]
                                    }
                                    cough_repo.create_cough_detection(db, detection_data)
                                    logging.info(f"Cough detection saved to database: {segment_timestamp}")
                                except Exception as db_error:
                                    logging.error(f"Failed to save cough detection to database: {db_error}")

                            try:
                                await websocket.send_text(json.dumps(event))
                            except Exception as e:
                                logging.info(f"Failed to send prediction event, client disconnected: {e}")
                                break
                    except Exception as e:
                        logging.error(f"Segment processing failed: {e}", exc_info=True)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        stream.stop_stream()
        stream.close()
        
        # Save the recorded audio
        if audio_buffer:
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(audio_buffer))
            wf.close()
            logging.info(f"Audio saved to {WAVE_OUTPUT_FILENAME}")


async def get_token(
    websocket: WebSocket,
    db: Session = Depends(get_db)
):
    token = websocket.query_params.get("token")
    if token is None:
        raise WebSocketDisconnect(code=403, reason="Token not provided")
    
    if TokenBlocklistRepo.is_token_blocklisted(db, token):
        raise WebSocketDisconnect(code=403, reason="Token has been blocklisted")

    decoded_token = JWTRepo.decode_token(token)
    if not decoded_token:
        raise WebSocketDisconnect(code=403, reason="Invalid token")
    
    return decoded_token

@router.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    """WebSocket endpoint for audio streaming."""
    try:
        token = await get_token(websocket, db)
        await audio_stream(websocket, token, db)
    except WebSocketDisconnect as e:
        logging.info(f"WebSocket disconnected: {e.reason}")
        # The connection is already closed by FastAPI, but if you need to send a custom message before,
        # you would do it before the exception is raised.
        # await websocket.close(code=e.code, reason=e.reason) - This is handled by FastAPI
        pass

@router.on_event("shutdown")
async def shutdown_event():
    # Terminate PyAudio on app shutdown
    if hasattr(router, 'pyaudio_instance'):
        router.pyaudio_instance.terminate()
        logging.info("PyAudio instance terminated.")

@router.get("/api/cough/detections")
async def list_cough_detections():
    items = []
    try:
        for j in sorted(router.media_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(j, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items.append(data)
            except Exception:
                pass
    except Exception:
        items = []
    return {"items": items}