import sys
from pathlib import Path
# Add parent directory to sys.path to access models package
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import engine
from tables import users as user_tables
from tables import cough_detections as cough_tables
from routes import users as user_routes
from routes import audio as audio_routes
from routes import video as video_routes
from routes import realtime_monitor
from routes import cough_stats


user_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")

# Configure CORS - Must be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to CareTaker API", "status": "active"}

# Include routers without prefix since we're handling it in the router
app.include_router(user_routes.router)
app.include_router(audio_routes.router)
app.include_router(video_routes.router)
app.include_router(realtime_monitor.router)
app.include_router(cough_stats.router)

# Mount static media directory
media_root = Path("media")
media_root.mkdir(parents=True, exist_ok=True)
(media_root / "cough").mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(media_root)), name="media")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=[".venv/"]
    )