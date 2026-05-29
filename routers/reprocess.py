from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from db import get_db
from models import WeddingImage, Celebration, FaceVector
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


@router.post("/all")
def reprocess_all(
    confirm: str = Query("", description="Must be 'yes' to actually queue. Otherwise dry-run."),
    only_legacy: bool = Query(
        True,
        description=(
            "When True, only re-process images whose face vectors were produced "
            "by an older embedding model. Set False to force-reprocess everything."
        ),
    ),
    db: Session = Depends(get_db),
):
    """Queue every image in the database for reprocessing.

    Designed for the one-time upgrade after a face-model change. Run as a
    dry-run first (``confirm=``) to see the count, then ``confirm=yes`` to fire.
    """
    target_model = settings.EMBEDDING_MODEL_VERSION

    if only_legacy:
        # An image is "legacy" if any of its face vectors are tagged with an
        # older model OR have no model tag at all (NULL = pre-versioning).
        legacy_image_ids_subq = (
            db.query(FaceVector.image_id)
            .filter(
                or_(
                    FaceVector.embedding_model.is_(None),
                    FaceVector.embedding_model != target_model,
                )
            )
            .distinct()
            .subquery()
        )
        images = db.query(WeddingImage).filter(WeddingImage.id.in_(legacy_image_ids_subq)).all()
    else:
        images = db.query(WeddingImage).all()

    if confirm.lower() != "yes":
        return {
            "dry_run": True,
            "would_queue": len(images),
            "target_embedding_model": target_model,
            "only_legacy": only_legacy,
            "message": (
                f"Dry-run: would queue {len(images)} images. "
                "Re-run with ?confirm=yes to actually queue."
            ),
        }

    queued = 0
    for img in images:
        # Reset the row so the dashboard's status surface shows movement.
        img.processed = "pending"
    db.commit()

    for img in images:
        dispatch_job("reprocess_image", image_id=str(img.id))
        queued += 1

    return {
        "queued": queued,
        "target_embedding_model": target_model,
        "only_legacy": only_legacy,
        "message": (
            f"Queued {queued} images for reprocessing via {settings.WORKER_BACKEND}. "
            "Use GET /reprocess/status to monitor progress."
        ),
    }


@router.get("/status")
def reprocess_status(db: Session = Depends(get_db)):
    """Quick status surface: how many images are pending/processing/completed/failed
    and how many face vectors match the current target embedding model."""
    target_model = settings.EMBEDDING_MODEL_VERSION

    by_status = {
        s: db.query(WeddingImage).filter(WeddingImage.processed == s).count()
        for s in ("pending", "processing", "completed", "failed")
    }
    total = sum(by_status.values())

    on_target = (
        db.query(FaceVector.image_id)
        .filter(FaceVector.embedding_model == target_model)
        .distinct()
        .count()
    )
    legacy = (
        db.query(FaceVector.image_id)
        .filter(
            or_(
                FaceVector.embedding_model.is_(None),
                FaceVector.embedding_model != target_model,
            )
        )
        .distinct()
        .count()
    )

    return {
        "target_embedding_model": target_model,
        "image_status": by_status,
        "image_total": total,
        "images_on_target_model": on_target,
        "images_with_legacy_model": legacy,
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
