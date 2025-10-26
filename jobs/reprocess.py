import redis
from db import SessionLocal
from routers.uploads import _process_image_faces
from models import WeddingImage
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_file_bytes(path: str):
    if path.startswith("http"):
        resp = requests.get(path)
        resp.raise_for_status()
        return resp.content
    with open(path, "rb") as f:
        return f.read()

def reprocess_image_job(image_id: str):
    """
    Called by the RQ worker to reprocess one image.
    """
    db = SessionLocal()
    img = db.query(WeddingImage).filter_by(id=image_id).first()
    if not img:
        logger.warning(f"⚠️ Image {image_id} not found in DB")
        return

    try:
        if img.processed in ("completed"):
            logger.info(f"⏭️ Skipping {img.filename} — already {img.processed}")
            return
        logger.info(f"♻️ Reprocessing {img.filename}")
        content = load_file_bytes(img.path)
        _process_image_faces(db, img, content)
        logger.info(f"✅ Completed reprocessing {img.filename}")
    except Exception as e:
        logger.exception(f"❌ Error reprocessing {img.filename}: {e}")
