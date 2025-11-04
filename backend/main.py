from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import engine
import tables.users as user_tables
import routes.users as user_routes
import routes.audio as audio_routes
import routes.video as video_routes


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=[".venv/"]
    )