from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from sqlalchemy import or_
from db import get_db
from models import WeddingImage, Celebration
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


@router.post("/{photographer}/{celebrant}")
def reprocess_celebration(
    photographer: str = Path(...),
    celebrant: str = Path(...),
    db: Session = Depends(get_db),
):
    """
    Queues all images for a specific celebration for reprocessing.
    Useful when model changes and you want to regenerate embeddings.
    """
    celebration = db.query(Celebration).filter(
        Celebration.photographer == photographer,
        Celebration.celebrant == celebrant,
    ).first()

    if not celebration:
        raise HTTPException(404, "Celebration not found")

    images = db.query(WeddingImage).filter(
        WeddingImage.celebration_id == celebration.id
    ).all()

    # Set all images to pending and commit BEFORE dispatching jobs
    # to avoid a race condition where Modal completes and sets "completed"
    # but this commit overwrites it back to "pending"
    for img in images:
        img.processed = "pending"
    db.commit()

    count = 0
    for img in images:
        job_id = dispatch_job("reprocess_image", image_id=str(img.id))
        count += 1
        logger.info(f"Queued job {job_id} for {img.filename}")

    return {
        "queued": count,
        "celebration_id": str(celebration.id),
        "message": f"Queued {count} images for reprocessing via {settings.WORKER_BACKEND}.",
    }
