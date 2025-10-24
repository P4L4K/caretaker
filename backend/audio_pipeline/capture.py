import pyaudio
import numpy as np
import noisereduce as nr
import config
import time

FORMAT_MAP = {
    "paInt16": pyaudio.paInt16,
    "paInt32": pyaudio.paInt32,
    "paFloat32": pyaudio.paFloat32
}

class LiveAudio:
    def __init__(self):
        self.CHUNK = config.AUDIO_CHUNK_SIZE
        self.RATE = config.AUDIO_RATE
        self.CHANNELS = config.AUDIO_CHANNELS
        self.FORMAT = FORMAT_MAP.get(config.AUDIO_FORMAT, pyaudio.paInt16)
        self.VAD_THRESHOLD = config.AUDIO_VAD_THRESHOLD
        self.MAX_SILENCE_CHUNKS = config.AUDIO_MAX_SILENCE_CHUNKS

        self.p = pyaudio.PyAudio()

        # Get default mic info
        self.default_device_info = self.p.get_default_input_device_info()
        device_name = self.default_device_info.get("name", "Unknown")
        print(f"\n🎤 Using microphone: {device_name}")
        print(f"   - Sample rate: {self.RATE}")
        print(f"   - Channels: {self.CHANNELS}")
        print(f"   - Chunk size: {self.CHUNK}\n")

        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        
        self.recording_started = False
        self.silent_chunks = 0
        self.metrics = []
        self.chunk_count = 0

    def read_chunk(self):
        """Read raw audio chunk, apply VAD & noise reduction."""
        try:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        except IOError as e:
            print(f"⚠️ Buffer overflow: {e}")
            return None, None

        np_data = np.frombuffer(data, dtype=np.int16).copy()
        rms = np.sqrt(np.mean(np_data.astype(np.float32)**2))
        is_voice = rms > self.VAD_THRESHOLD
        self.chunk_count += 1

        # Noise reduction
        if is_voice:
            cleaned = nr.reduce_noise(y=np_data.astype(np.float32), sr=self.RATE)
            cleaned = np.clip(cleaned, -32768, 32767).astype(np.int16)
            self.recording_started = True
            self.silent_chunks = 0
        else:
            cleaned = np_data
            if self.recording_started:
                self.silent_chunks += 1
                if self.silent_chunks > self.MAX_SILENCE_CHUNKS:
                    self.stop()

        # Store metrics
        metric = {
            "chunk": self.chunk_count,
            "rms": float(rms),
            "is_voice": bool(is_voice),
            "silent_chunks": self.silent_chunks,
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.metrics.append(metric)

        # --- 👀 Console feedback ---
        status = "🎙️ Voice" if is_voice else "🤫 Silence"
        print(f"[{metric['timestamp']}] Chunk {self.chunk_count}: RMS={rms:.2f} → {status}")

        return cleaned, metric

    def stop(self):
        """Stop audio stream safely."""
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print(f"\n🛑 Audio stream stopped. Total chunks processed: {self.chunk_count}\n")
