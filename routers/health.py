from fastapi import APIRouter
from datetime import datetime
from services import face_service

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": face_service is not None,
    }
