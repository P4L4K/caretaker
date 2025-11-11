"""
Multiprocessed model inference for CareTaker system.
Runs audio, video, and emotion recognition models in parallel with GPU support.
"""

import multiprocessing as mp
import time
import logging
from typing import Dict, Any
import numpy as np
import cv2
import torch
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {MODEL_DEVICE}")

class ModelManager:
    def __init__(self):
        self.audio_model = None
        self.video_model = None
        self.emotion_model = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize all models with GPU support if available."""
        try:
            # Import models here to avoid circular imports
            from models.audio.cough.infer import AudioDetector
            from models.video.fall_detection import FallDetector
            from models.video.emotion_recognition import get_emotion_from_frame
            
            # Initialize models
            self.audio_model = AudioDetector(device=MODEL_DEVICE)
            self.video_model = FallDetector()
            if MODEL_DEVICE == 'cuda':
                self.video_model.model = self.video_model.model.to(MODEL_DEVICE)
            
            # Warm up models
            self.warmup_models()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def warmup_models(self):
        """Warm up models with dummy data."""
        logger.info("Warming up models...")
        try:
            # Warm up audio model
            dummy_audio = np.random.randn(16000).astype(np.float32)
            _ = self.audio_model.detect(dummy_audio)
            
            # Warm up video model
            dummy_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            _ = self.video_model.detect_fall(dummy_frame)
            
            # Warm up emotion model
            _ = self.video_model.emotion_model.predict(dummy_frame)
            
            logger.info("Models warmed up successfully")
        except Exception as e:
            logger.warning(f"Warning during model warmup: {e}")

def audio_worker(input_queue: mp.Queue, output_queue: mp.Queue):
    """Process audio frames for cough detection."""
    model = ModelManager()
    while True:
        try:
            if not input_queue.empty():
                audio_data = input_queue.get()
                if audio_data is None:  # Sentinel value to stop the process
                    break
                
                # Process audio
                result = model.audio_model.detect(audio_data)
                output_queue.put({"type": "audio", "data": result})
                
        except Exception as e:
            logger.error(f"Error in audio worker: {e}")
            time.sleep(0.1)

def video_worker(input_queue: mp.Queue, output_queue: mp.Queue):
    """Process video frames for fall detection."""
    model = ModelManager()
    while True:
        try:
            if not input_queue.empty():
                frame_data = input_queue.get()
                if frame_data is None:  # Sentinel value to stop the process
                    break
                
                # Process video frame
                result = model.video_model.detect_fall(frame_data)
                output_queue.put({"type": "fall", "data": result})
                
        except Exception as e:
            logger.error(f"Error in video worker: {e}")
            time.sleep(0.1)

def emotion_worker(input_queue: mp.Queue, output_queue: mp.Queue):
    """Process video frames for emotion recognition."""
    model = ModelManager()
    while True:
        try:
            if not input_queue.empty():
                frame_data = input_queue.get()
                if frame_data is None:  # Sentinel value to stop the process
                    break
                
                # Process emotion
                result = model.emotion_model.predict(frame_data)
                output_queue.put({"type": "emotion", "data": result})
                
        except Exception as e:
            logger.error(f"Error in emotion worker: {e}")
            time.sleep(0.1)

class MultiModelProcessor:
    def __init__(self):
        # Queues for inter-process communication
        self.audio_queue = mp.Queue()
        self.video_queue = mp.Queue()
        self.emotion_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # Initialize processes
        self.processes = [
            mp.Process(target=audio_worker, args=(self.audio_queue, self.result_queue)),
            mp.Process(target=video_worker, args=(self.video_queue, self.result_queue)),
            mp.Process(target=emotion_worker, args=(self.emotion_queue, self.result_queue))
        ]
        
        # Start processes
        for p in self.processes:
            p.daemon = True
            p.start()
    
    def process_audio(self, audio_data: np.ndarray):
        """Add audio data to the processing queue."""
        self.audio_queue.put(audio_data)
    
    def process_video(self, frame: np.ndarray):
        """Add video frame to the processing queue."""
        self.video_queue.put(frame)
    
    def process_emotion(self, frame: np.ndarray):
        """Add frame for emotion analysis to the queue."""
        self.emotion_queue.put(frame)
    
    def get_results(self):
        """Get all available results from the result queue."""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results
    
    def stop(self):
        """Stop all worker processes."""
        # Send sentinel values to stop workers
        self.audio_queue.put(None)
        self.video_queue.put(None)
        self.emotion_queue.put(None)
        
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

# WebSocket endpoint for real-time processing
app = FastAPI()
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    processor = MultiModelProcessor()
    logger.info("MultiModelProcessor started with GPU support")

@app.on_event("shutdown")
async def shutdown_event():
    if processor:
        processor.stop()
        logger.info("MultiModelProcessor stopped")

@app.websocket("/ws/process")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_bytes()
            
            # Parse the data (assuming it's a JSON with type and data)
            try:
                message = json.loads(data.decode('utf-8'))
                data_type = message.get('type')
                payload = message.get('data')
                
                if data_type == 'audio':
                    # Process audio data
                    audio_data = np.frombuffer(payload, dtype=np.float32)
                    processor.process_audio(audio_data)
                elif data_type == 'video':
                    # Process video frame (assuming it's a JPEG-encoded frame)
                    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    processor.process_video(frame)
                
                # Get and send results
                results = processor.get_results()
                if results:
                    await websocket.send_json({"results": results})
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

def main():
    """Run the FastAPI server with WebSocket support."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Set TensorFlow to use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("TensorFlow GPU configured")
        except RuntimeError as e:
            logger.error(f"Error configuring TensorFlow GPU: {e}")
    
    # Start the server
    main()
