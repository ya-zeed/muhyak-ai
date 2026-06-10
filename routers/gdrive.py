import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import Celebration
from jobs.dispatcher import dispatch_job
from services import redis_client
from services.gdrive import list_folder_images
from config import settings

logger = logging.getLogger("routers.gdrive")

router = APIRouter(prefix="/gdrive", tags=["gdrive"])


class ImportRequest(BaseModel):
    photographer: str
    celebrant: str
    folder_id: str
    api_key: str


def _progress_key(celebration_id: str, field: str) -> str:
    return f"gdrive_import:{celebration_id}:{field}"


@router.post("/import")
def start_import(
    req: ImportRequest,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Kick off a background import of every image in a public Drive folder.

    Returns immediately; the work runs on the configured worker backend and
    progress is polled via GET /gdrive/import/status.
    """
    celebration = db.query(Celebration).filter(
        Celebration.photographer == req.photographer,
        Celebration.celebrant == req.celebrant,
    ).first()
    if not celebration:
        raise HTTPException(404, "Celebration not found")

    try:
        files = list_folder_images(req.folder_id, req.api_key)
    except Exception:
        logger.exception("Drive folder listing failed")
        raise HTTPException(400, "تعذر الوصول للمجلد. تأكد أن المجلد عام (Public)")

    cid = str(celebration.id)
    total = len(files)

    try:
        redis_client.set(_progress_key(cid, "total"), total, ex=86400)
        redis_client.set(_progress_key(cid, "done"), 0, ex=86400)
        redis_client.set(_progress_key(cid, "failed"), 0, ex=86400)
    except Exception:
        logger.warning("Could not init gdrive import progress", exc_info=True)

    def _dispatch_all():
        for f in files:
            try:
                dispatch_job(
                    "import_drive_image",
                    file_id=f["id"],
                    api_key=req.api_key,
                    filename=f.get("name", "image.jpg"),
                    mime_type=f.get("mimeType", "image/jpeg"),
                    celebrant=req.celebrant,
                    photographer=req.photographer,
                    celebration_id=cid,
                )
            except Exception:
                logger.exception("Failed to dispatch import job")

    background.add_task(_dispatch_all)

    return {
        "queued": total,
        "celebration_id": cid,
        "message": f"Queued {total} images for import via {settings.WORKER_BACKEND}.",
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
