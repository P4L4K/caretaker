import asyncio
import json
import logging
import numpy as np
import pyaudio
import noisereduce as nr
import wave
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from scipy.io.wavfile import write as write_wav

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
    # Initialize PyAudio on app startup
    router.pyaudio_instance = pyaudio.PyAudio()
    logging.info("PyAudio instance created.")

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

async def audio_stream(websocket: WebSocket, token: dict):
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

from ..repository.users import JWTRepo
from ..repository.token_blocklist import TokenBlocklistRepo
from ..config import get_db
from sqlalchemy.orm import Session
from fastapi import Depends, Query


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
        await audio_stream(websocket, token)
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