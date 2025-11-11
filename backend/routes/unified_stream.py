"""
Unified WebSocket endpoint for simultaneous audio and video streaming.
Handles real-time cough detection, emotion recognition, and fall detection concurrently.
"""

import asyncio
import json
import logging
import numpy as np
import pyaudio
import noisereduce as nr
import cv2
import base64
from datetime import datetime
from pathlib import Path
import soundfile as sf
import librosa
import tensorflow as tf
import joblib
import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from typing import Optional

from tables.users import CareTaker, CareRecipient
from repository.users import UsersRepo, JWTRepo
from repository.token_blocklist import TokenBlocklistRepo
from repository import cough_detections as cough_repo
from config import get_db
from models.video.emotion_recognition import get_emotion_from_frame
from models.video.fall_detection import FallDetector

# Import enhancement function from audio.py
import sys
sys.path.append(str(Path(__file__).parent))
from audio import enhance_audio_for_elderly

try:
    from models.video.fall_detection import FallDetector
    FALL_DETECTION_AVAILABLE = True
except Exception as e:
    logging.warning(f"Fall detection not available: {e}")
    FALL_DETECTION_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

# Audio constants - HIGHLY OPTIMIZED FOR ELDERLY CARE
CHUNK = 4096
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
NOISE_GATE_THRESHOLD = -55  # VERY sensitive for weak elderly voices
SILENCE_THRESHOLD_RMS = 300  # MUCH lower for very quiet elderly sounds
ATTACK_TIME = 0.003  # Faster to catch weak transients
RELEASE_TIME = 0.25  # Longer to capture complete weak coughs
HOLD_TIME = 0.15  # Much longer hold for weak elderly speech

router = APIRouter()

# Global model storage
_models = {
    'cough_classifier': None,
    'yamnet_model': None,
    'preprocessor': None,
    'fall_detector': None,
    'yamnet_hub_model': None
}

@router.on_event("startup")
async def startup_event():
    """Load all models on startup"""
    logging.info("Loading unified stream models...")
    
    # Load audio models
    backend_dir = Path(__file__).parent.parent
    models_dir = backend_dir.parent / "models" / "audio" / "cough" / "models"
    preproc_dir = backend_dir.parent / "models" / "audio" / "cough" / "preprocessor"
    
    try:
        from keras.models import load_model as keras_load_model
        _models['cough_classifier'] = keras_load_model(str(models_dir / "yamnet_88.keras"))
        logging.info("Cough classifier loaded!")
    except Exception as e:
        try:
            _models['cough_classifier'] = tf.keras.models.load_model(str(models_dir / "yamnet_88.keras"), compile=False)
            logging.info("Cough classifier loaded (TF method)!")
        except Exception as e2:
            logging.error(f"Failed to load cough classifier: {e} | {e2}")
    
    try:
        _models['yamnet_model'] = tf.saved_model.load(str(models_dir / "yamnet_88_savedmodel"))
        logging.info("YAMNet model loaded!")
    except Exception as e:
        logging.error(f"Failed to load YAMNet: {e}")
    
    try:
        _models['preprocessor'] = joblib.load(str(preproc_dir / "preprocessor_saved.pkl"))
        logging.info("Preprocessor loaded!")
    except Exception as e:
        logging.error(f"Failed to load preprocessor: {e}")
    
    # Load fall detector
    if FALL_DETECTION_AVAILABLE:
        try:
            _models['fall_detector'] = FallDetector()
            logging.info("Fall detector loaded!")
        except Exception as e:
            logging.error(f"Failed to load fall detector: {e}")
    
    # Media directory
    router.media_dir = Path("media/cough")
    router.media_dir.mkdir(parents=True, exist_ok=True)
    router.target_sr = 16000
    router.target_duration = 4.68
    router.max_segments = 30
    router.threshold = 0.5


def calculate_db(rms):
    """Convert RMS to decibels"""
    if rms <= 0:
        return -100
    return 20 * np.log10(rms / 32768.0)


def normalize_audio(audio, target_db=-20.0):
    """Normalize audio to target dB level"""
    rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
    if rms < 1e-6:
        return audio
    
    current_db = 20 * np.log10(rms / 32768.0)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20.0)
    
    normalized = audio.astype(np.float32) * gain_linear
    normalized = np.clip(normalized, -32768, 32767)
    
    return normalized.astype(np.int16)


def apply_noise_gate_with_hold(audio_data, rms, gate_state, hold_counter, 
                                attack_samples, release_samples, hold_samples, adaptive_threshold):
    """Apply noise gate with attack, hold, and release envelope"""
    db_level = calculate_db(rms)
    gate_open = db_level > NOISE_GATE_THRESHOLD and rms > adaptive_threshold
    
    attack_samples = max(1, attack_samples)
    release_samples = max(1, release_samples)
    
    if gate_open:
        gate_state = min(1.0, gate_state + 1.0 / attack_samples)
        hold_counter = hold_samples
    elif hold_counter > 0:
        gate_state = 1.0
        hold_counter -= 1
    elif gate_state > 0.0:
        gate_state = max(0.0, gate_state - 1.0 / release_samples)
    
    gated_audio = audio_data * gate_state
    return gated_audio.astype(np.int16), gate_state, hold_counter


async def process_audio_chunk(audio_data, meta_dict):
    """Process audio chunk for cough detection"""
    try:
        tmp_path = router.media_dir / "_tmp_segment.wav"
        data_i16 = np.frombuffer(audio_data, dtype=np.int16)
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
        y_enhanced = enhance_audio_for_elderly(y, sr=router.target_sr)

        if _models['cough_classifier'] is None or _models['preprocessor'] is None:
            return None

        waveform = tf.convert_to_tensor(y_enhanced, dtype=tf.float32)
        
        # Load YAMNet from TF Hub on-the-fly
        import tensorflow_hub as hub
        if _models['yamnet_hub_model'] is None:
            logging.info("Loading YAMNet from TensorFlow Hub...")
            _models['yamnet_hub_model'] = hub.load('https://tfhub.dev/google/yamnet/1')
        
        _, embeddings, _ = _models['yamnet_hub_model'](waveform)
        embeddings = embeddings.numpy()
        
        if embeddings.shape[0] < router.max_segments:
            embeddings = np.pad(embeddings, ((0, router.max_segments - embeddings.shape[0]), (0, 0)), mode='constant')
        else:
            embeddings = embeddings[:router.max_segments]

        meta_df = pd.DataFrame([meta_dict])
        meta_input = _models['preprocessor'].transform(meta_df).astype('float32')
        pred = _models['cough_classifier'].predict([embeddings[np.newaxis, :, :], meta_input], verbose=0)
        prob = float(pred[0, 0])
        label = "Cough" if prob >= router.threshold else "Not Cough"
        
        return {"probability": prob, "label": label, "y": y, "sr": sr}
    except Exception as e:
        logging.error(f"Audio processing error: {e}", exc_info=True)
        return None


async def process_video_frame(frame):
    """Process video frame for emotion and fall detection"""
    try:
        result = {}
        
        # Emotion detection
        dom, conf, _ = get_emotion_from_frame(
            frame,
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            target_width=640,
        )
        result['emotion'] = dom or "neutral"
        result['emotion_confidence'] = conf
        
        # Fall detection
        fall_res = {"fall_detected": False, "timestamp": None}
        if FALL_DETECTION_AVAILABLE and _models['fall_detector']:
            fall_res = _models['fall_detector'].detect_fall(frame)
        
        result['fall_detected'] = bool(fall_res.get("fall_detected"))
        result['fall_timestamp'] = fall_res.get("timestamp")
        
        return result
    except Exception as e:
        logging.error(f"Video processing error: {e}", exc_info=True)
        return None


async def unified_stream_handler(websocket: WebSocket, token: dict, db: Session):
    """Handle unified audio and video streaming"""
    p = pyaudio.PyAudio()
    
    # Extract user metadata
    username = token.get("sub")
    caretaker = UsersRepo.find_by_username(db, CareTaker, username) if username else None
    recipient = None
    if caretaker and caretaker.care_recipients:
        recipient = caretaker.care_recipients[0]
    
    meta_dict = {
        "age": recipient.age if recipient else 30,
        "gender": recipient.gender.value if recipient and recipient.gender else "Male",
        "respiratory_condition": recipient.respiratory_condition_status if recipient else False
    }
    
    logging.info(f"Unified stream started for user: {username}")
    
    # Setup audio stream
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    
    await websocket.accept()
    logging.info("Unified WebSocket connection accepted")
    
    # Audio processing state
    audio_buffer = []
    gate_state = 0.0
    hold_counter = 0
    segment_active = False
    segment_bytes = []
    segment_started_at = None
    max_segment_duration = 5.0
    
    attack_samples = max(1, int(ATTACK_TIME * RATE / CHUNK))
    release_samples = max(1, int(RELEASE_TIME * RATE / CHUNK))
    hold_samples = max(1, int(HOLD_TIME * RATE / CHUNK))
    
    noise_floor_buffer = []
    noise_floor_size = 100
    MIN_NOISE_THRESHOLD = 50   # VERY low baseline for extremely quiet elderly voices
    MAX_NOISE_THRESHOLD = 500  # Lower cap to stay sensitive to weak sounds
    
    # Video processing state
    video_frame_buffer = None
    last_video_process_time = datetime.now()
    video_process_interval = 1.5  # Process video every 1.5 seconds
    
    try:
        while True:
            # Check connection
            try:
                if websocket.client_state.name != "CONNECTED":
                    logging.info("WebSocket disconnected")
                    break
            except Exception:
                logging.info("WebSocket connection check failed")
                break
            
            # === AUDIO PROCESSING ===
            data = stream.read(CHUNK, exception_on_overflow=False)
            data_np = np.frombuffer(data, dtype=np.int16)
            
            rms_raw = np.sqrt(np.mean(np.square(data_np.astype(np.float32))))
            
            # Adaptive noise floor - HIGHLY SENSITIVE FOR ELDERLY
            if rms_raw < SILENCE_THRESHOLD_RMS * 0.5:  # Only track very quiet ambient noise
                if len(noise_floor_buffer) >= noise_floor_size:
                    noise_floor_buffer.pop(0)
                noise_floor_buffer.append(rms_raw)
            
            if len(noise_floor_buffer) > 20:
                adaptive_threshold = np.clip(
                    np.percentile(noise_floor_buffer, 75) * 1.2,  # Lower percentile and multiplier
                    MIN_NOISE_THRESHOLD,
                    MAX_NOISE_THRESHOLD
                )
            else:
                adaptive_threshold = MIN_NOISE_THRESHOLD
            
            # Noise reduction - VERY GENTLE for weak elderly sounds
            reduced_noise = nr.reduce_noise(
                y=data_np, 
                sr=RATE, 
                stationary=True,
                prop_decrease=0.3  # Much less aggressive
            )
            
            rms = np.sqrt(np.mean(np.square(reduced_noise.astype(np.float32))))
            
            # Apply noise gate
            gated_audio, gate_state, hold_counter = apply_noise_gate_with_hold(
                reduced_noise, rms, gate_state, hold_counter,
                attack_samples, release_samples, hold_samples, adaptive_threshold
            )
            
            db_level = calculate_db(rms)
            
            # Check segment duration
            should_process_segment = False
            if segment_active and segment_started_at:
                elapsed = (datetime.utcnow() - segment_started_at).total_seconds()
                if elapsed >= max_segment_duration:
                    should_process_segment = True
            
            # Collect audio data
            if gate_state > 0.1:
                audio_buffer.append(gated_audio.tobytes())
                # Save ORIGINAL RAW audio (before noise reduction) for playback
                segment_bytes.append(data_np.tobytes())
                if not segment_active:
                    segment_active = True
                    segment_started_at = datetime.utcnow()
                
                # VISUALIZATION THRESHOLD: Only show waveform if gate is substantially open
                display_threshold = 0.3  # Show waveform only when gate is 30% or more open
                
                if gate_state >= display_threshold:
                    display_waveform = gated_audio.tolist()
                else:
                    display_waveform = (gated_audio * 0.2).astype(np.int16).tolist()
                
                # Send audio waveform data
                try:
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "waveform": display_waveform,
                        "rms": float(rms),
                        "db": float(db_level),
                        "gate_open": gate_state > 0.5,
                        "gate_level": float(gate_state),
                        "hold_active": hold_counter > 0
                    }))
                except Exception as e:
                    logging.info(f"Failed to send audio data: {e}")
                    break
            
            # Process audio segment when gate closes or max duration reached
            if (gate_state <= 0.1 or should_process_segment) and segment_active and segment_bytes:
                raw_bytes = b"".join(segment_bytes)
                segment_timestamp = segment_started_at if segment_started_at else datetime.utcnow()
                segment_active = False
                segment_bytes = []
                segment_started_at = None
                duration_sec = len(raw_bytes) / (2 * RATE)
                
                if duration_sec >= 0.3:
                    # Process audio in background
                    audio_result = await process_audio_chunk(raw_bytes, meta_dict)
                    
                    if audio_result and audio_result['label'] == 'Cough':
                        prob = audio_result['probability']
                        y = audio_result['y']
                        sr = audio_result['sr']
                        
                        # Save cough audio
                        ts = segment_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
                        out_path = router.media_dir / f"cough_{ts}.wav"
                        y_int16 = (y * 32768).astype(np.int16)
                        y_normalized = normalize_audio(y_int16, target_db=-20.0)
                        sf.write(str(out_path), y_normalized, sr, subtype='PCM_16')
                        
                        event = {
                            "type": "cough_detection",
                            "timestamp": segment_timestamp.isoformat() + "Z",
                            "probability": prob,
                            "label": "Cough",
                            "media_url": f"/media/cough/{out_path.name}"
                        }
                        
                        # Save metadata
                        sidecar = {
                            "timestamp": event["timestamp"],
                            "probability": prob,
                            "label": "Cough",
                            "media_url": event["media_url"],
                            "username": username,
                            "caretaker_id": caretaker.id if caretaker else None,
                            "recipient_id": recipient.id if recipient else None,
                            "age": meta_dict["age"],
                            "gender": meta_dict["gender"],
                            "respiratory_condition": meta_dict["respiratory_condition"]
                        }
                        with open(str(out_path.with_suffix('.json')), 'w', encoding='utf-8') as f:
                            json.dump(sidecar, f)
                        
                        # Save to database
                        try:
                            detection_data = {
                                "timestamp": segment_timestamp,
                                "probability": prob,
                                "label": "Cough",
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
                            logging.info(f"Failed to send cough detection: {e}")
                            break
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.001)
    
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"Unified stream error: {e}", exc_info=True)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def get_token(websocket: WebSocket, db: Session = Depends(get_db)):
    """Validate WebSocket token"""
    token = websocket.query_params.get("token")
    if token is None:
        raise WebSocketDisconnect(code=403, reason="Token not provided")
    
    if TokenBlocklistRepo.is_token_blocklisted(db, token):
        raise WebSocketDisconnect(code=403, reason="Token has been blocklisted")

    decoded_token = JWTRepo.decode_token(token)
    if not decoded_token:
        raise WebSocketDisconnect(code=403, reason="Invalid token")
    
    return decoded_token


@router.websocket("/ws/unified")
async def unified_websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    """Unified WebSocket endpoint for simultaneous audio and video streaming"""
    try:
        token = await get_token(websocket, db)
        await unified_stream_handler(websocket, token, db)
    except WebSocketDisconnect as e:
        logging.info(f"WebSocket disconnected: {e.reason}")


@router.post("/video-frame")
async def receive_video_frame(websocket: WebSocket):
    """Receive video frames from frontend (alternative to embedding in WebSocket)"""
    # This endpoint can be used if you want to send video frames separately
    pass
