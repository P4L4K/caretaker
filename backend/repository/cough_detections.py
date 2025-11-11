from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from tables.cough_detections import CoughDetection
from datetime import datetime, timedelta


def create_cough_detection(db: Session, detection_data: dict):
    """Create a new cough detection record"""
    detection = CoughDetection(**detection_data)
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def get_all_detections(db: Session, limit: int = 100):
    """Get all cough detections ordered by timestamp descending"""
    return db.query(CoughDetection).order_by(CoughDetection.timestamp.desc()).limit(limit).all()


def get_detections_by_user(db: Session, username: str, limit: int = 100):
    """Get cough detections for a specific user"""
    return db.query(CoughDetection).filter(
        CoughDetection.username == username
    ).order_by(CoughDetection.timestamp.desc()).limit(limit).all()


def get_detections_by_date_range(db: Session, start_date: datetime, end_date: datetime):
    """Get cough detections within a date range"""
    return db.query(CoughDetection).filter(
        and_(
            CoughDetection.timestamp >= start_date,
            CoughDetection.timestamp <= end_date
        )
    ).order_by(CoughDetection.timestamp.desc()).all()


def get_total_detections(db: Session):
    """Get total count of all cough detections"""
    return db.query(func.count(CoughDetection.id)).scalar()


def get_today_detections(db: Session):
    """Get count of today's cough detections"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    return db.query(func.count(CoughDetection.id)).filter(
        and_(
            CoughDetection.timestamp >= today_start,
            CoughDetection.timestamp < today_end
        )
    ).scalar()


def get_last_detection(db: Session):
    """Get the most recent cough detection"""
    return db.query(CoughDetection).order_by(CoughDetection.timestamp.desc()).first()


def get_recent_detections(db: Session, limit: int = 10):
    """Get recent cough detections"""
    return db.query(CoughDetection).order_by(CoughDetection.timestamp.desc()).limit(limit).all()


def get_detection_stats(db: Session):
    """Get comprehensive statistics about cough detections"""
    total = get_total_detections(db)
    today = get_today_detections(db)
    last_detection = get_last_detection(db)
    recent = get_recent_detections(db, limit=10)
    
    # Calculate average probability
    avg_prob = db.query(func.avg(CoughDetection.probability)).scalar() or 0.0
    
    return {
        "total_detections": total or 0,
        "today_detections": today or 0,
        "last_detection_time": last_detection.timestamp.isoformat() + "Z" if last_detection else None,
        "average_probability": float(avg_prob),
        "recent_detections": [
            {
                "id": d.id,
                "timestamp": d.timestamp.isoformat() + "Z",
                "probability": d.probability,
                "label": d.label,
                "media_url": d.media_url,
                "username": d.username
            }
            for d in recent
        ]
    }


def delete_old_detections(db: Session, days_old: int = 30):
    """Delete cough detections older than specified days"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    deleted = db.query(CoughDetection).filter(CoughDetection.timestamp < cutoff_date).delete()
    db.commit()
    return deleted
