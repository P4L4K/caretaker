from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .config import engine
from .tables import users as user_tables
from .routes import users as user_routes
from .routes import audio as audio_routes
from .routes import video as video_routes


user_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to CareTaker API", "status": "active"}

# Include routers without prefix since we're handling it in the router
app.include_router(user_routes.router)
app.include_router(audio_routes.router)
app.include_router(video_routes.router)

# Mount static media directory
media_root = Path("media")
media_root.mkdir(parents=True, exist_ok=True)
(media_root / "cough").mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(media_root)), name="media")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=[".venv/"]
    )