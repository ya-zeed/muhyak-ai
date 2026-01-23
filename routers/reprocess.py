from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import or_
from db import get_db
from models import WeddingImage
from jobs.dispatcher import dispatch_job
from config import settings
import logging

logger = logging.getLogger("routers.reprocess")

router = APIRouter(prefix="/reprocess", tags=["reprocess"])


@router.post("/unprocessed")
def reprocess_unprocessed(db: Session = Depends(get_db)):
    """
    Queues all unprocessed images for reprocessing using configured backend.
    """
    images = (
        db.query(WeddingImage)
        .filter(
            or_(
                WeddingImage.processed != "completed",
                WeddingImage.processed.is_(None),
            )
        )
        .all()
    )

    count = 0
    for img in images:
        job_id = dispatch_job("reprocess_image", image_id=str(img.id))
        count += 1
        logger.info(f"Queued job {job_id} for {img.filename}")

    return {
        "queued": count,
        "message": f"Queued {count} images for background reprocessing via {settings.WORKER_BACKEND}.",
    }
