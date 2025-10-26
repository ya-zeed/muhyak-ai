from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import or_
import redis
from rq import Queue
from db import get_db
from models import WeddingImage
from jobs.reprocess import reprocess_image_job
import os
import logging

logger = logging.getLogger("routers.reprocess")

router = APIRouter(prefix="/reprocess", tags=["reprocess"])

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
conn = redis.from_url(redis_url)
queue = Queue("default", connection=conn)

@router.post("/unprocessed")
def reprocess_unprocessed(db: Session = Depends(get_db)):
    """
    🧾 Queues all unprocessed images for reprocessing using Redis RQ.
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
        job = queue.enqueue(reprocess_image_job, img.id)
        count += 1
        logger.info(f"Queued job {job.id} for {img.filename}")

    return {
        "queued": count,
        "message": f"Queued {count} images for background reprocessing.",
    }
