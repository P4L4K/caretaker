from fastapi import APIRouter
from audio_session import AudioSessionManager

router = APIRouter(prefix="/audio", tags=["audio"])
audio_manager = AudioSessionManager()

@router.post("/start")
def start_audio_session(user_id: int):
    """Start live audio session for logged-in user."""
    session = audio_manager.start_session(user_id)
    return {"status": "started", "metrics_len": len(session.metrics)}

@router.post("/stop")
def stop_audio_session(user_id: int):
    """Stop live audio session."""
    audio_manager.stop_session(user_id)
    return {"status": "stopped"}
