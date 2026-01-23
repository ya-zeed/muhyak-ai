from fastapi import APIRouter
from datetime import datetime
from config import settings

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "worker_backend": settings.WORKER_BACKEND,
    }
