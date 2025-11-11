from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config import Base
import datetime


class CoughDetection(Base):
    __tablename__ = "cough_detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    probability = Column(Float, nullable=False)
    label = Column(String, nullable=False)
    media_url = Column(String, nullable=True)
    
    # User information
    username = Column(String, nullable=True)
    caretaker_id = Column(Integer, ForeignKey("caretakers.id", ondelete="SET NULL"), nullable=True)
    recipient_id = Column(Integer, ForeignKey("care_recipients.id", ondelete="SET NULL"), nullable=True)
    
    # Metadata
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    respiratory_condition = Column(String, nullable=True)
    
    # Tracking
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    caretaker = relationship("CareTaker", foreign_keys=[caretaker_id])
    recipient = relationship("CareRecipient", foreign_keys=[recipient_id])
