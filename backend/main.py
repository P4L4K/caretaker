from fastapi import FastAPI
from config import engine
import tables.users as user_tables

user_tables.Base.metadata.create_all(bind=engine)

app = FastAPI()

# @app.get("/")
# async def root():
#     return "Hello, CareTaker!"

