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
import time
from datetime import datetime
from pathlib import Path
import soundfile as sf
import librosa
import tensorflow as tf
import joblib
import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List, Tuple

# Import from local modules
from .audio import (
    AudioConfig, audio_config, 
    process_audio_chunk,
    calculate_db,
    normalize_audio,
    get_audio_stream,
    enhance_audio_for_elderly
)
from .video import (
    get_emotion_from_frame,
    FALL_DETECTION_AVAILABLE,
    _fall_detector,
    _VideoCaptureThread
)

# Import FallDetector directly to ensure it's available
try:
    from models.video.fall_detection import FallDetector
    _fall_detector = FallDetector()
    FALL_DETECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Fall detection not available: {e}")
    _fall_detector = None
    FALL_DETECTION_AVAILABLE = False

# Import from other modules
from tables.users import CareTaker, CareRecipient
from repository.users import UsersRepo, JWTRepo
from repository.token_blocklist import TokenBlocklistRepo
from repository import cough_detections as cough_repo
from config import get_db

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)

# Create a structured JSON log file with timestamp
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f'detections_{session_id}.jsonl'  # JSON Lines format

class DetectionLogger:
    _instance = None
    
    def __new__(cls, log_file: Path):
        if cls._instance is None:
            cls._instance = super(DetectionLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_file: Path):
        if self._initialized:
            return
            
        self.log_file = log_file
        self._initialized = True
        
        print(f"[DEBUG] Initializing logger with file: {log_file}")
        print(f"[DEBUG] Log directory exists: {log_file.parent.exists()}")
        print(f"[DEBUG] Log directory writable: {os.access(str(log_file.parent), os.W_OK)}")
        
        # Create a dedicated logger for detections
        self.logger = logging.getLogger('caretaker.detections')
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # Ensure the log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        try:
            file_handler = logging.FileHandler(str(log_file.absolute()), mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)
            print(f"[DEBUG] File handler added to logger")
        except Exception as e:
            print(f"[ERROR] Failed to create file handler: {e}")
            
        # Also log to console for debugging
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console)
        
        # Test logging
        self.logger.info("Logger initialized successfully")
        print(f"[DEBUG] Logger initialization complete")
        
        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            # File handler for JSON logs
            try:
                file_handler = logging.FileHandler(
                    str(log_file.absolute()),
                    mode='a',
                    encoding='utf-8',
                    delay=False
                )
                file_handler.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(file_handler)
                
                # Console handler for human-readable logs
                console = logging.StreamHandler(sys.stdout)
                console.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                self.logger.addHandler(console)
                
                # Log the start of the session
                self.logger.info(json.dumps({
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'level': 'INFO',
                    'message': f'Detection logging started',
                    'session_id': session_id,
                    'log_file': str(log_file.absolute())
                }, default=str))
                
            except Exception as e:
                # Fallback to console if file logging fails
                print(f"Failed to initialize file logging: {e}", file=sys.stderr)
                console = logging.StreamHandler()
                console.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(console)
                self.logger.error(f"Failed to initialize file logging: {e}")
        
        # Make sure the file is writable
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('')  # Test write
        except Exception as e:
            self.logger.error(f"Log file is not writable: {e}")
            raise
    
    def log_detection(
        self,
        detection_type: str,  # 'cough', 'fall', 'emotion'
        prediction: Dict[str, Any],
        user_id: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        # Debug log to confirm the method is being called
        print(f"[DEBUG] Logging detection - Type: {detection_type}, User: {user_id}")
        print(f"[DEBUG] Log file: {self.log_file}")
        print(f"[DEBUG] Handlers: {self.logger.handlers}")
        try:
            print(f"[DEBUG] Log file exists: {self.log_file.exists()}")
            print(f"[DEBUG] Log file writable: {os.access(str(self.log_file), os.W_OK)}")
        except Exception as e:
            print(f"[DEBUG] Error checking log file: {e}")
        """
        Log a detection event with structured data.
        
        Args:
            detection_type: Type of detection ('cough', 'fall', 'emotion')
            prediction: Dictionary containing prediction details
            user_id: Optional user ID
            processing_time_ms: Optional processing time in milliseconds
            metadata: Additional metadata to include in the log
        """
        try:
            # Create log entry with all required fields
            log_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'session_id': session_id,
                'detection_type': detection_type,
                'level': 'INFO',
                'user_id': user_id,
                'processing_time_ms': round(processing_time_ms, 3) if processing_time_ms is not None else None,
                'prediction': prediction,
                'metadata': metadata or {}
            }
            
            # Convert to JSON string
            log_line = json.dumps(log_entry, default=str)
            
            # Log to file and console
            self.logger.info(log_line)
            
            # Force flush to ensure logs are written immediately
            for handler in self.logger.handlers:
                handler.flush()
                
        except Exception as e:
            # If JSON serialization fails, log the error with traceback
            error_msg = f"Failed to log detection: {str(e)}"
            print(f"ERROR: {error_msg}", file=sys.stderr)
            self.logger.error(error_msg, exc_info=True)
            
            # Try to log a simplified version
            try:
                safe_entry = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'session_id': session_id,
                    'level': 'ERROR',
                    'error': 'Failed to serialize detection',
                    'detection_type': str(detection_type)[:100],
                    'user_id': str(user_id)[:100] if user_id else None
                }
                self.logger.error(json.dumps(safe_entry))
            except:
                # Last resort - log a simple message
                self.logger.error(f"[{datetime.utcnow().isoformat()}Z] Failed to log detection")

# Initialize the detection logger
print("[DEBUG] Initializing detection logger...")
detection_logger = DetectionLogger(log_file)
print("[DEBUG] Detection logger initialized")

# Configure basic logging for other purposes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Use audio configuration from audio.py
audio_cfg = AudioConfig()

# Video capture thread
_video_capture = _VideoCaptureThread()

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


async def process_audio_chunk(audio_data, meta_dict):
    """Process audio chunk for cough detection"""
    try:
        tmp_path = router.media_dir / "_tmp_segment.wav"
        data_i16 = np.frombuffer(audio_data, dtype=np.int16)
        sf.write(str(tmp_path), data_i16.astype(np.int16), audio_cfg.RATE, subtype='PCM_16')

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
        print("[DEBUG] Starting emotion detection...")
        try:
            dom, conf, all_emotions = get_emotion_from_frame(
                frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=True,
                target_width=640,
            )
            print(f"[DEBUG] Emotion detection result - emotion: {dom}, confidence: {conf}")
            if all_emotions:
                print(f"[DEBUG] All emotions: {all_emotions}")
            
            result['emotion'] = dom or "neutral"
            result['emotion_confidence'] = float(conf) if conf is not None else 0.0
            result['all_emotions'] = all_emotions
            
        except Exception as e:
            print(f"[ERROR] Error in emotion detection: {str(e)}")
            import traceback
            traceback.print_exc()
            result['emotion'] = "error"
            result['emotion_confidence'] = 0.0
            result['emotion_error'] = str(e)
        
        # Log emotion detection if confidence is above threshold (lowered to 0.1 to capture more events)
        emotion_confidence_threshold = 0.1  # Lowered threshold to capture more emotion detections
        print(f"[DEBUG] Checking emotion confidence: {conf} (threshold: {emotion_confidence_threshold})")
        if conf is not None and conf >= emotion_confidence_threshold:
            print(f"[DEBUG] Emotion detected: {dom} (confidence: {conf:.2f})")
            try:
                detection_logger.log_detection(
                    detection_type="emotion",
                    prediction={
                        "emotion": dom,
                        "confidence": float(conf),
                        "all_emotions": _  # This would contain the full emotion distribution if available
                    },
                    user_id=None,  # Add user ID if available
                    processing_time_ms=0,  # Add actual processing time if available
                    metadata={
                        "timestamp": datetime.utcnow().isoformat() + 'Z',
                        "detection_type": "emotion"
                    }
                )
                print("[DEBUG] Emotion detection logged successfully")
            except Exception as e:
                print(f"[ERROR] Failed to log emotion detection: {e}")
                import traceback
                traceback.print_exc()
        
        # Fall detection
        fall_res = {"fall_detected": False, "timestamp": None}
        if FALL_DETECTION_AVAILABLE and _models['fall_detector']:
            print("[DEBUG] Running fall detection...")
            fall_res = _models['fall_detector'].detect_fall(frame)
            print(f"[DEBUG] Fall detection result: {fall_res}")
        else:
            print(f"[DEBUG] Fall detection not available or not loaded. Available: {FALL_DETECTION_AVAILABLE}, Detector: {'loaded' if _models.get('fall_detector') else 'not loaded'}")
        
        result['fall_detected'] = bool(fall_res.get("fall_detected"))
        result['fall_timestamp'] = fall_res.get("timestamp")
        
        # Log fall detection if detected
        if result['fall_detected']:
            print(f"[DEBUG] Fall detected! Logging detection...")
            try:
                detection_logger.log_detection(
                    detection_type="fall",
                    prediction={"confidence": 1.0, "angle": fall_res.get("angle", 0)},
                    user_id=None,  # Add user ID if available
                    processing_time_ms=0,  # Add actual processing time if available
                    metadata={"timestamp": str(fall_res.get("timestamp"))}
                )
                print("[DEBUG] Fall detection logged successfully")
            except Exception as e:
                print(f"[ERROR] Failed to log fall detection: {e}")
                import traceback
                traceback.print_exc()
        
        return result
    except Exception as e:
        error_msg = f"Video processing error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        logging.error(error_msg, exc_info=True)
        
        # Log the error to the detection log as well
        try:
            detection_logger.log_detection(
                detection_type="error",
                prediction={"error": str(e), "type": "video_processing"},
                metadata={
                    "timestamp": datetime.utcnow().isoformat() + 'Z',
                    "error_type": "video_processing"
                }
            )
        except Exception as log_error:
            print(f"[ERROR] Failed to log video processing error: {log_error}")
            
        return None


async def unified_stream_handler(websocket: WebSocket, token: dict, db: Session):
    """Handle unified audio and video streaming"""
    print("[HANDLER] Starting unified stream handler")
    
    # Set to track if we're shutting down
    is_shutting_down = False
    
    # Initialize PyAudio
    print("[AUDIO] Initializing PyAudio...")
    p = None
    try:
        p = pyaudio.PyAudio()
        print("[AUDIO] PyAudio initialized")
    except Exception as e:
        print(f"[AUDIO] Failed to initialize PyAudio: {e}")
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return
    
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
    
    # Initialize audio stream
    print(f"[AUDIO] Opening audio stream with format={audio_cfg.FORMAT}, channels={audio_cfg.CHANNELS}, rate={audio_cfg.RATE}, chunk={audio_cfg.CHUNK}")
    try:
        stream = p.open(
            format=audio_cfg.FORMAT,
            channels=audio_cfg.CHANNELS,
            rate=audio_cfg.RATE,
            input=True,
            frames_per_buffer=audio_cfg.CHUNK
        )
        print("[AUDIO] Audio stream opened successfully")
    except Exception as e:
        print(f"[AUDIO] Failed to open audio stream: {e}")
        import traceback
        traceback.print_exc()
        raise
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
    
    attack_samples = max(1, int(ATTACK_TIME * audio_cfg.RATE / audio_cfg.CHUNK))
    release_samples = max(1, int(RELEASE_TIME * audio_cfg.RATE / audio_cfg.CHUNK))
    hold_samples = max(1, int(HOLD_TIME * audio_cfg.RATE / audio_cfg.CHUNK))
    
    noise_floor_buffer = []
    noise_floor_size = 100
    MIN_NOISE_THRESHOLD = 50   # VERY low baseline for extremely quiet elderly voices
    MAX_NOISE_THRESHOLD = 500  # Lower cap to stay sensitive to weak sounds
    
    # Video processing state
    video_frame_buffer = None
    last_video_process_time = datetime.now()
    video_process_interval = 1.5  # Process video every 1.5 seconds
    # Main processing loop
    print("[HANDLER] Entering main processing loop")
    loop_count = 0
    
    try:
        while True:
            loop_count += 1
            if loop_count % 100 == 0:  # Log every 100 iterations to avoid flooding
                print(f"[HANDLER] Processing loop iteration {loop_count}")
            
            # Check for incoming WebSocket messages
            try:
                # Set a short timeout to avoid blocking
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                try:
                    message = json.loads(data)
                    print(f"[WEBSOCKET] Received message: {message}")
                    
                    # Handle handshake message
                    if message.get('type') == 'handshake':
                        print(f"[WEBSOCKET] Handshake received from client: {message.get('message')}")
                        await websocket.send_json({
                            'type': 'handshake_ack',
                            'message': 'Server acknowledged',
                            'timestamp': datetime.utcnow().isoformat() + 'Z'
                        })
                        continue
                        
                except json.JSONDecodeError:
                    print(f"[WEBSOCKET] Received non-JSON message: {data}")
                    
            except asyncio.TimeoutError:
                # No message received, continue with audio processing
                pass
            except Exception as e:
                print(f"[WEBSOCKET] Error receiving message: {e}")
                break
            
            # Check if WebSocket is still connected
            try:
                if websocket.client_state.name != "CONNECTED":
                    print("[HANDLER] WebSocket disconnected, stopping stream")
                    logging.info("WebSocket disconnected, stopping stream")
                    break
            except Exception as e:
                print(f"[HANDLER] WebSocket connection check failed: {e}")
                logging.error(f"WebSocket connection check failed: {e}")
                break
            # Process audio data
            try:
                data = stream.read(audio_cfg.CHUNK, exception_on_overflow=False)
                # Send audio data back to client for visualization
                if loop_count % 10 == 0:  # Don't flood the WebSocket
                    await websocket.send_json({
                        'type': 'audio_data',
                        'timestamp': datetime.utcnow().isoformat() + 'Z'
                    })
            except Exception as e:
                print(f"[AUDIO] Error reading from audio stream: {e}")
                break
                data_np = np.frombuffer(data, dtype=np.int16)
                
                rms_raw = np.sqrt(np.mean(np.square(data_np.astype(np.float32))))
                
                # Adaptive noise floor - HIGHLY SENSITIVE FOR ELDERLY
                if rms_raw < audio_cfg.SILENCE_THRESHOLD_RMS * 0.5:  # Only track very quiet ambient noise
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
            try:
                reduced_noise = nr.reduce_noise(
                    y=data_np, 
                    sr=audio_cfg.RATE, 
                    stationary=True,
                    prop_decrease=0.3  # Much less aggressive
                )
                
                rms = np.sqrt(np.mean(np.square(reduced_noise.astype(np.float32))))
            except Exception as e:
                print(f"[AUDIO] Error in noise reduction: {e}")
                rms = rms_raw  # Fall back to raw RMS if noise reduction fails
            
            # Apply noise gate
            try:
                gated_audio, gate_state, hold_counter = apply_noise_gate_with_hold(
                    reduced_audio=reduced_noise,
                    rms=rms,
                    gate_state=gate_state,
                    hold_counter=hold_counter,
                    attack_samples=attack_samples,
                    release_samples=release_samples,
                    hold_samples=hold_samples,
                    adaptive_threshold=adaptive_threshold,
                    audio_config=audio_cfg
                )
            except Exception as e:
                print(f"[AUDIO] Error in noise gate: {e}")
                gated_audio = reduced_noise  # Bypass noise gate on error
            
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
                duration_sec = len(raw_bytes) / (2 * audio_cfg.RATE)
                
                if duration_sec >= 0.3:
                    # Process audio in background with timing
                    print("[AUDIO] Processing audio chunk...")
                    start_time = time.time()
                    try:
                        audio_result = await process_audio_chunk(raw_bytes, meta_dict)
                        processing_time = (time.time() - start_time) * 1000
                        print(f"[AUDIO] Processing completed in {processing_time:.2f}ms")
                        
                        # Send detection result to client
                        if audio_result and 'label' in audio_result:
                            await websocket.send_json({
                                'type': 'detection',
                                'event': 'prediction',
                                'label': audio_result['label'],
                                'probability': float(audio_result.get('probability', 0.0)),
                                'timestamp': datetime.utcnow().isoformat() + 'Z',
                                'processing_time_ms': processing_time
                            })
                            
                    except Exception as e:
                        print(f"[AUDIO] Error processing audio: {e}")
                        import traceback
                        traceback.print_exc()
                        audio_result = None
                        processing_time = 0
                    
                    if audio_result and 'label' in audio_result:
                        print(f"[AUDIO] Audio result: {audio_result}")
                        if audio_result['label'] == 'Cough':
                            print(f"[AUDIO] Cough detected! Logging for user {username}")
                            try:
                                detection_logger.log_detection(
                                    detection_type="cough",
                                    prediction={"label": audio_result['label'], "probability": float(audio_result.get('probability', 0.0))},
                                    user_id=username,
                                    processing_time_ms=processing_time,
                                    metadata={"media_url": ""}
                                )
                                print("[AUDIO] Successfully logged cough detection")
                            except Exception as e:
                                print(f"[AUDIO] Failed to log detection: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"[AUDIO] No cough detected (label: {audio_result['label']})")
                    else:
                        print("[AUDIO] No valid audio result to process")
                    
                    if audio_result and audio_result['label'] == 'Cough':
                        prob = audio_result['probability']
                        y = audio_result.get('y')
                        sr = audio_result.get('sr')
                        
                        # Save cough audio if we have the raw audio data
                        if y is not None and sr is not None:
                            try:
                                ts = segment_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
                                out_path = router.media_dir / f"cough_{ts}.wav"
                                y_int16 = (y * 32768).astype(np.int16)
                                
                                # Ensure media directory exists
                                os.makedirs(router.media_dir, exist_ok=True)
                                
                                # Save the audio file
                                sf.write(str(out_path), y_int16, int(sr), 'PCM_16')
                                print(f"[AUDIO] Saved cough audio to {out_path}")
                                
                                # Update the detection with the media URL
                                media_url = f"/media/{out_path.name}"
                                
                            except Exception as e:
                                print(f"[AUDIO] Error saving cough audio: {e}")
                                media_url = ""
                        else:
                            media_url = ""
                            print("[AUDIO] No audio data to save for this detection")
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
    
    except asyncio.CancelledError:
        # Handle graceful shutdown
        is_shutting_down = True
        print("[HANDLER] Received cancellation signal, shutting down...")
        raise
    except WebSocketDisconnect:
        if not is_shutting_down:
            logging.info("WebSocket disconnected by client")
    except Exception as e:
        if not is_shutting_down:
            logging.error(f"Unified stream error: {e}", exc_info=True)
    finally:
        # Cleanup resources
        print("[HANDLER] Cleaning up resources...")
        try:
            if 'stream' in locals() and stream.is_active():
                stream.stop_stream()
                stream.close()
                print("[AUDIO] Audio stream closed")
        except Exception as e:
            print(f"[ERROR] Error closing audio stream: {e}")
            
        try:
            if 'p' in locals():
                p.terminate()
                print("[AUDIO] PyAudio terminated")
        except Exception as e:
            print(f"[ERROR] Error terminating PyAudio: {e}")
            
        print("[HANDLER] Cleanup complete")


async def get_token(websocket: WebSocket, db: Session = Depends(get_db)):
    """Validate WebSocket token"""
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        logger.info(f"Received token: {token}")
        
        if not token:
            logger.error("No token provided")
            raise WebSocketDisconnect(code=403, reason="Token not provided")
        
        # Check if token is blocklisted
        if TokenBlocklistRepo.is_token_blocklisted(db, token):
            logger.error("Token is blocklisted")
            raise WebSocketDisconnect(code=403, reason="Token has been blocklisted")

        # Decode and validate token
        decoded_token = JWTRepo.decode_token(token)
        logger.info(f"Decoded token: {decoded_token}")
        
        if not decoded_token:
            logger.error("Failed to decode token")
            raise WebSocketDisconnect(code=403, reason="Invalid token")
        
        return decoded_token
        
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}", exc_info=True)
        raise WebSocketDisconnect(code=403, reason=f"Token validation failed: {str(e)}")



@router.post("/video-frame")
async def receive_video_frame(websocket: WebSocket):
    """Receive video frames from frontend (alternative to embedding in WebSocket)"""
    # This endpoint can be used if you want to send video frames separately
    pass
