from fastapi import APIRouter
from datetime import datetime
from services import face_service

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Wedding Face Search API", "version": "1.0.0"}

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": face_service is not None,
    }
