 
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
from ..tables.users import CareTaker, CareRecipient
from ..repository.users import UsersRepo
from ..repository.users import JWTRepo
from ..repository.token_blocklist import TokenBlocklistRepo
from ..config import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
CHUNK = 4096
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
WAVE_OUTPUT_FILENAME = "output.wav"

# Noise gate parameters - OPTIMIZED FOR ELDERLY CARE ENVIRONMENTS
# Designed for low-medium noise (hospitals, nursing homes, bedrooms)
# Target sounds: coughs, snores, groans, breathing irregularities, falls, calls for help
NOISE_GATE_THRESHOLD = -45  # dB threshold (sensitive for elderly voices)
SILENCE_THRESHOLD_RMS = 600  # Lower for quieter elderly speech and sounds
# CRITICAL: Fast attack for transient detection (coughs, falls)
ATTACK_TIME = 0.005  # 5ms - Captures sudden sounds like coughs, falls
# Moderate release for natural decay
RELEASE_TIME = 0.15  # 150ms - Slightly longer to capture full coughs/snores
# Hold time to prevent choppy gating during labored breathing
HOLD_TIME = 0.08  # 80ms - Longer hold for elderly speech patterns

router = APIRouter()
p = pyaudio.PyAudio()

@router.on_event("startup")
async def startup_event():
    router.pyaudio_instance = pyaudio.PyAudio()
    logging.info("PyAudio instance created.")
    models_dir = Path("models/audio/cough/models")
    preproc_dir = Path("models/audio/cough/preprocessor")
    router.cough_classifier = None
    try:
        router.cough_classifier = keras_load_model(str(models_dir / "yamnet_88.keras"))
    except Exception as e1:
        try:
            router.cough_classifier = tf.keras.models.load_model(str(models_dir / "yamnet_88.keras"), compile=False)
        except Exception as e2:
            logging.error(f"Failed to load cough classifier: {e1} | {e2}")
    try:
        router.yamnet_model = tf.saved_model.load(str(models_dir / "yamnet_88_savedmodel"))
    except Exception:
        router.yamnet_model = None
        logging.error("Failed to load YAMNet saved model.")
    try:
        router.preprocessor = joblib.load(str(preproc_dir / "preprocessor_saved.pkl"))
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
    MIN_NOISE_THRESHOLD = 100  # Lower baseline for quiet elderly care settings
    MAX_NOISE_THRESHOLD = 800  # Cap to prevent adaptation to loud events
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            data_np = np.frombuffer(data, dtype=np.int16)
            
            # Calculate initial RMS
            rms_raw = np.sqrt(np.mean(np.square(data_np.astype(np.float32))))
            
            # Adaptive noise floor estimation - OPTIMIZED FOR CARE ENVIRONMENTS
            # Only update noise floor with LOW values (ambient noise, not patient sounds)
            if rms_raw < SILENCE_THRESHOLD_RMS * 0.7:  # Only track quiet ambient noise
                if len(noise_floor_buffer) >= noise_floor_size:
                    noise_floor_buffer.pop(0)
                noise_floor_buffer.append(rms_raw)
            
            # Calculate adaptive threshold from ambient noise only
            if len(noise_floor_buffer) > 20:
                adaptive_threshold = np.clip(
                    np.percentile(noise_floor_buffer, 85) * 1.3,
                    MIN_NOISE_THRESHOLD,
                    MAX_NOISE_THRESHOLD
                )
            else:
                adaptive_threshold = MIN_NOISE_THRESHOLD
            
            # Always apply noise reduction for consistent processing
            # Gentler noise reduction to preserve elderly speech characteristics
            reduced_noise = nr.reduce_noise(
                y=data_np, 
                sr=RATE, 
                stationary=True,
                prop_decrease=0.5  # Less aggressive to preserve weak sounds
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
            
            # Only save and send if gate is significantly open
            if gate_state > 0.1:
                audio_buffer.append(gated_audio.tobytes())
                segment_bytes.append(gated_audio.tobytes())
                if not segment_active:
                    segment_active = True
                    segment_started_at = datetime.utcnow()
                
                # Send waveform data to the frontend
                await websocket.send_text(json.dumps({
                    "waveform": gated_audio.tolist(),
                    "rms": float(rms),
                    "db": float(db_level),
                    "gate_open": gate_state > 0.5,
                    "gate_level": float(gate_state),
                    "hold_active": hold_counter > 0
                }))
            else:
                # Send silence indicator
                await websocket.send_text(json.dumps({
                    "waveform": [0] * len(gated_audio),
                    "rms": 0.0,
                    "db": -100.0,
                    "gate_open": False,
                    "gate_level": 0.0,
                    "hold_active": False
                }))

                if segment_active and segment_bytes:
                    try:
                        raw_bytes = b"".join(segment_bytes)
                        segment_active = False
                        segment_bytes = []
                        duration_sec = len(raw_bytes) / (2 * RATE)
                        if duration_sec >= 0.3:
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

                            if router.yamnet_model is None or router.cough_classifier is None or router.preprocessor is None:
                                logging.warning("Skipping prediction: model or preprocessor not loaded.")
                                continue

                            waveform = tf.convert_to_tensor(y, dtype=tf.float32)
                            _, embeddings, _ = router.yamnet_model(waveform)
                            embeddings = embeddings.numpy()
                            if embeddings.shape[0] < router.max_segments:
                                embeddings = np.pad(embeddings, ((0, router.max_segments - embeddings.shape[0]), (0, 0)), mode='constant')
                            else:
                                embeddings = embeddings[:router.max_segments]

                            meta_df = pd.DataFrame([meta_dict])
                            meta_input = router.preprocessor.transform(meta_df).astype('float32')
                            pred = router.cough_classifier.predict([embeddings[np.newaxis, :, :], meta_input], verbose=0)
                            prob = float(pred[0, 0])
                            label = "Cough" if prob >= router.threshold else "Not Cough"

                            event = {
                                "event": "prediction",
                                "timestamp": segment_started_at.isoformat() + "Z",
                                "probability": prob,
                                "label": label
                            }

                            if label == "Cough":
                                ts = segment_started_at.strftime("%Y%m%dT%H%M%S%fZ")
                                out_path = router.media_dir / f"cough_{ts}.wav"
                                sf.write(str(out_path), y, sr, subtype='PCM_16')
                                event["media_url"] = f"/media/cough/{out_path.name}"
                                sidecar = {"timestamp": event["timestamp"], "probability": prob, "label": label, "media_url": event["media_url"], "username": username, "caretaker_id": (caretaker.id if caretaker else None), "recipient_id": (recipient.id if recipient else None), "age": meta_dict["age"], "gender": meta_dict["gender"], "respiratory_condition": meta_dict["respiratory_condition"]}
                                with open(str(out_path.with_suffix('.json')), 'w', encoding='utf-8') as f:
                                    json.dump(sidecar, f)

                            await websocket.send_text(json.dumps(event))
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