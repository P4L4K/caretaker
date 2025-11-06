from sqlalchemy import Column, Integer, String
from ..config import Base

class TokenBlocklist(Base):
    __tablename__ = "token_blocklist"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True)
