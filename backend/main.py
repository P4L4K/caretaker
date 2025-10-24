from fastapi import FastAPI
from config import engine
import tables.users as user_tables
import routes.users as user_routes
import routes.audio as audio_routes
user_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")

app.include_router(user_routes.router)
# Audio route
app.include_router(audio_routes.router)
# @app.get("/")
# async def root():
#     return "Hello, CareTaker!"




