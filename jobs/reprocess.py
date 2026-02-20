from db import SessionLocal
from routers.uploads import _process_image_faces
from models import WeddingImage, FaceVector
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
    Deletes old face vectors and regenerates with current model.
    """
    db = SessionLocal()
    try:
        img = db.query(WeddingImage).filter_by(id=image_id).first()
        if not img:
            logger.warning(f"⚠️ Image {image_id} not found in DB")
            return

        logger.info(f"♻️ Reprocessing {img.filename}")

        # Delete old face vectors
        deleted = db.query(FaceVector).filter(FaceVector.image_id == img.id).delete()
        logger.info(f"🗑️ Deleted {deleted} old face vectors")

        # Always use the original image so bbox coordinates match what the frontend displays
        file_url = img.file_path
        content = load_file_bytes(file_url)

        _process_image_faces(db, img, content)
        logger.info(f"✅ Completed reprocessing {img.filename}")
    except Exception as e:
        logger.exception(f"❌ Error reprocessing {img.filename}: {e}")
    finally:
        db.close()
