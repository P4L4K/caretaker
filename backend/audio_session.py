from audio_pipeline.capture import LiveAudio
from audio_pipeline.preprocess import preprocess_audio_chunk
import threading
import time

class AudioSessionManager:
    def __init__(self):
        self.sessions = {}

    def start_session(self, user_id: int):
        if user_id in self.sessions:
            print(f"⚠️ Session for user {user_id} already active.")
            return self.sessions[user_id]

        print(f"\n🚀 Starting audio session for user {user_id}...\n")
        live_audio = LiveAudio()
        self.sessions[user_id] = live_audio

        def run_audio():
            start_time = time.time()
            while live_audio.stream.is_active():
                chunk, metrics = live_audio.read_chunk()
                if chunk is None:
                    continue
                _ = preprocess_audio_chunk(chunk)
                time.sleep(0.01)
            duration = time.time() - start_time
            print(f"✅ User {user_id}'s session ended after {duration:.2f} seconds.\n")

        thread = threading.Thread(target=run_audio, daemon=True)
        thread.start()

        return live_audio

    def stop_session(self, user_id: int):
        if user_id in self.sessions:
            print(f"🧹 Stopping audio session for user {user_id}...")
            self.sessions[user_id].stop()
            del self.sessions[user_id]
