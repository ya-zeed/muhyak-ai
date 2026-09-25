import logging
from collections import Counter

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import Celebration, WeddingImage
from jobs.dispatcher import dispatch_job
from services import redis_client
from services.gphotos import is_album_url, list_album_images
from config import settings

logger = logging.getLogger("routers.gphotos")

router = APIRouter(prefix="/gphotos", tags=["gphotos"])


class ImportRequest(BaseModel):
    photographer: str
    celebrant: str
    album_url: str


def _progress_key(celebration_id: str, field: str) -> str:
    return f"gphotos_import:{celebration_id}:{field}"


@router.post("/import")
def start_import(
    req: ImportRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Kick off a background import of every photo in a shared Google Photos
    album. Same flow as /gdrive/import: returns immediately, one worker job per
    photo, progress polled via GET /gphotos/import/status.
    """
    if not is_album_url(req.album_url):
        raise HTTPException(400, "رابط ألبوم Google Photos غير صالح")

    celebration = db.query(Celebration).filter(
        Celebration.photographer == req.photographer,
        Celebration.celebrant == req.celebrant,
    ).first()
    if not celebration:
        raise HTTPException(404, "Celebration not found")

    try:
        photos = list_album_images(req.album_url)
    except Exception:
        logger.exception("Google Photos album listing failed")
        raise HTTPException(400, "تعذر الوصول للألبوم. تأكد أن الألبوم مشترك برابط")

    cid = str(celebration.id)

    # Filenames are stable per photo (derived from Google's media key), so a
    # re-run skips what already landed and only retries the failures.
    remaining = Counter(
        name
        for (name,) in db.query(WeddingImage.filename)
        .filter(WeddingImage.celebration_id == celebration.id)
        .all()
    )
    pending = []
    for p in photos:
        if remaining[p["name"]] > 0:
            remaining[p["name"]] -= 1
            continue
        pending.append(p)
    skipped = len(photos) - len(pending)
    total = len(pending)

    try:
        redis_client.set(_progress_key(cid, "total"), total, ex=86400)
        redis_client.set(_progress_key(cid, "done"), 0, ex=86400)
        redis_client.set(_progress_key(cid, "failed"), 0, ex=86400)
    except Exception:
        logger.warning("Could not init gphotos import progress", exc_info=True)

    def _dispatch_all():
        for p in pending:
            try:
                dispatch_job(
                    "import_gphotos_image",
                    url=p["url"],
                    filename=p["name"],
                    celebrant=req.celebrant,
                    photographer=req.photographer,
                    celebration_id=cid,
                )
            except Exception:
                logger.exception("Failed to dispatch gphotos import job")

    background.add_task(_dispatch_all)

    return {
        "queued": total,
        "skipped": skipped,
        "celebration_id": cid,
        "message": (
            f"Queued {total} photos ({skipped} already imported, skipped) "
            f"via {settings.WORKER_BACKEND}."
        ),
    }


@router.get("/import/status")
def import_status(celebration_id: str = Query(...)):
    def _read(field: str) -> int:
        try:
            v = redis_client.get(_progress_key(celebration_id, field))
            return int(v) if v is not None else 0
        except Exception:
            return 0

    return {
        "total": _read("total"),
        "done": _read("done"),
        "failed": _read("failed"),
    }
