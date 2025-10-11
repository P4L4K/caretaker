from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL ="postgresql://postgres:start12@localhost:5432/caretaker"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db=SessionLocal()
    try: 
        yield db
        print("DB session successful")
    except Exception as e:
        print("DB session failed")
    finally: 
        db.close()

#jwt
SECRET_KEY = "crie"
ALGORITHM ='HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 30
