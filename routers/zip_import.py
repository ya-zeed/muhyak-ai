"""Import every image inside an uploaded .zip archive.

Built for the cloud-folder use case: a user downloads a shared OneDrive / Google
Drive folder as a single ZIP (one click in the web UI, no account/API needed) and
hands it to muhyak. We stream the upload to a temp file, then in the background
extract each image, compress it to the platform's display size (reusing the same
`compress_image` as the Drive importer), and queue it through the standard
`process_image` pipeline — so it works unchanged on both the RQ and Modal
backends with no worker redeploy.
"""
from __future__ import annotations

import os
import logging
import tempfile
import zipfile

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy.orm import Session

from db import get_db
from models import Celebration
from jobs.dispatcher import dispatch_job
from services import redis_client
from services.gdrive import compress_image
from config import settings

logger = logging.getLogger("routers.zip_import")

router = APIRouter(prefix="/upload", tags=["zip"])

IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff",
)


def _progress_key(celebration_id: str, field: str) -> str:
    return f"zip_import:{celebration_id}:{field}"


def _is_image_entry(info: zipfile.ZipInfo) -> bool:
    """True for real image files, skipping directories, dotfiles and the
    __MACOSX/ resource-fork junk that macOS adds to archives."""
    name = info.filename
    if info.is_dir():
        return False
    if name.startswith("__MACOSX/") or os.path.basename(name).startswith("."):
        return False
    return name.lower().endswith(IMAGE_EXTS)


@router.post("/zip")
async def upload_zip(
    background: BackgroundTasks,
    photographer: str = Form(...),
    celebrant: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Queue every image in the uploaded ZIP for import.

    Returns immediately with the number of images found; compression and
    queuing run in the background and progress is polled via
    GET /upload/zip/status.
    """
    celebration = db.query(Celebration).filter(
        Celebration.photographer == photographer,
        Celebration.celebrant == celebrant,
    ).first()
    if not celebration:
        raise HTTPException(404, "Celebration not found")

    # Stream the upload to a temp file so a multi-GB archive never sits in memory.
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = tmp.name
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()

        try:
            with zipfile.ZipFile(tmp_path) as zf:
                names = [i.filename for i in zf.infolist() if _is_image_entry(i)]
        except zipfile.BadZipFile:
            raise HTTPException(400, "الملف ليس أرشيف ZIP صالح")

        total = len(names)
        if total == 0:
            raise HTTPException(400, "لا توجد صور في الملف المضغوط")

        cid = str(celebration.id)
        try:
            redis_client.set(_progress_key(cid, "total"), total, ex=86400)
            redis_client.set(_progress_key(cid, "done"), 0, ex=86400)
            redis_client.set(_progress_key(cid, "failed"), 0, ex=86400)
        except Exception:
            logger.warning("Could not init zip import progress", exc_info=True)

        background.add_task(
            _process_zip, tmp_path, names, cid, celebrant, photographer
        )
        # Ownership of tmp_path handed to the background task; don't delete here.
        tmp_path = None

        return {
            "queued": total,
            "celebration_id": cid,
            "message": f"Queued {total} images from zip via {settings.WORKER_BACKEND}.",
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _process_zip(
    zip_path: str,
    names: list[str],
    celebration_id: str,
    celebrant: str,
    photographer: str,
) -> None:
    """Extract → compress → dispatch each image, one at a time (low memory)."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in names:
                try:
                    raw = zf.read(name)
                    compressed = compress_image(raw)
                    out_name = os.path.basename(name).rsplit(".", 1)[0] + ".jpg"
                    dispatch_job(
                        "process_image",
                        celebrant=celebrant,
                        photographer=photographer,
                        filename=out_name,
                        content=compressed,
                        celebration_id=celebration_id,
                    )
                    _incr(celebration_id, "done")
                except Exception:
                    logger.exception("Failed to import %s from zip", name)
                    _incr(celebration_id, "failed")
                    _incr(celebration_id, "done")
        logger.info("✅ Finished queuing %d images from zip", len(names))
    finally:
        try:
            os.unlink(zip_path)
        except OSError:
            pass


def _incr(celebration_id: str, field: str) -> None:
    try:
        key = _progress_key(celebration_id, field)
        redis_client.incr(key)
        redis_client.expire(key, 86400)
    except Exception:
        logger.warning("zip import progress update failed", exc_info=True)


@router.get("/zip/status")
def zip_status(celebration_id: str = Query(...)):
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
