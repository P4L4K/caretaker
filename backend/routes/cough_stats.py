from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from config import get_db
from repository import cough_detections as cough_repo
from datetime import datetime

router = APIRouter(prefix="/api/cough", tags=["Cough Statistics"])


@router.get("/stats")
async def get_cough_statistics(db: Session = Depends(get_db)):
    """
    Get comprehensive cough detection statistics including:
    - Total detections
    - Today's detections
    - Last detection time
    - Average probability
    - Recent detections
    """
    try:
        stats = cough_repo.get_detection_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")


@router.get("/detections/all")
async def get_all_cough_detections(limit: int = 100, db: Session = Depends(get_db)):
    """Get all cough detections with optional limit"""
    try:
        detections = cough_repo.get_all_detections(db, limit=limit)
        return {
            "items": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat() + "Z",
                    "probability": d.probability,
                    "label": d.label,
                    "media_url": d.media_url,
                    "username": d.username,
                    "age": d.age,
                    "gender": d.gender,
                    "respiratory_condition": d.respiratory_condition
                }
                for d in detections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching detections: {str(e)}")


@router.get("/detections/user/{username}")
async def get_user_cough_detections(username: str, limit: int = 100, db: Session = Depends(get_db)):
    """Get cough detections for a specific user"""
    try:
        detections = cough_repo.get_detections_by_user(db, username, limit=limit)
        return {
            "items": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat() + "Z",
                    "probability": d.probability,
                    "label": d.label,
                    "media_url": d.media_url,
                    "username": d.username
                }
                for d in detections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user detections: {str(e)}")


@router.get("/detections/range")
async def get_detections_by_date_range(
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db)
):
    """Get cough detections within a date range (ISO format: YYYY-MM-DD)"""
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        detections = cough_repo.get_detections_by_date_range(db, start, end)
        return {
            "items": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat() + "Z",
                    "probability": d.probability,
                    "label": d.label,
                    "media_url": d.media_url,
                    "username": d.username
                }
                for d in detections
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching detections: {str(e)}")


@router.get("/total")
async def get_total_cough_count(db: Session = Depends(get_db)):
    """Get total count of all cough detections"""
    try:
        total = cough_repo.get_total_detections(db)
        return {"total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching total: {str(e)}")


@router.get("/today")
async def get_today_cough_count(db: Session = Depends(get_db)):
    """Get count of today's cough detections"""
    try:
        today = cough_repo.get_today_detections(db)
        return {"today": today}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching today's count: {str(e)}")


@router.get("/last")
async def get_last_cough_detection(db: Session = Depends(get_db)):
    """Get the most recent cough detection"""
    try:
        last = cough_repo.get_last_detection(db)
        if last:
            return {
                "id": last.id,
                "timestamp": last.timestamp.isoformat() + "Z",
                "probability": last.probability,
                "label": last.label,
                "media_url": last.media_url,
                "username": last.username
            }
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching last detection: {str(e)}")
