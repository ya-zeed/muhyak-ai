"""RQ worker for importing a single Drive image (local/dev backend).

Mirrors modal_worker.import_drive_image. Stores the full-res original as
file_path and a downscaled JPEG as compressed_file_path; faces are detected on
the compressed image so bbox coordinates match what the gallery displays.
"""
import logging
import uuid

from db import SessionLocal
from models import WeddingImage, FaceVector
from utils import load_image_from_bytes, calculate_file_hash
from config import settings
from services import face_service, upload_to_s3, redis_client
from services.gdrive import download_drive_file, compress_image

logger = logging.getLogger(__name__)


def _progress_incr(celebration_id: str, failed: bool = False) -> None:
    try:
        redis_client.incr(f"gdrive_import:{celebration_id}:done")
        redis_client.expire(f"gdrive_import:{celebration_id}:done", 86400)
        if failed:
            redis_client.incr(f"gdrive_import:{celebration_id}:failed")
            redis_client.expire(f"gdrive_import:{celebration_id}:failed", 86400)
    except Exception:
        logger.warning("failed to update gdrive import progress", exc_info=True)


def import_drive_image_job(
    file_id: str,
    api_key: str,
    filename: str,
    mime_type: str,
    celebrant: str,
    photographer: str,
    celebration_id: str,
) -> None:
    db = SessionLocal()
    try:
        raw = download_drive_file(file_id, api_key)
        file_hash = calculate_file_hash(raw)

        existing = db.query(WeddingImage).filter(
            WeddingImage.file_hash == file_hash
        ).first()
        if existing:
            logger.info(f"🟡 Skipped duplicate {filename}")
            _progress_incr(celebration_id)
            return

        compressed = compress_image(raw)
        out_name = filename.rsplit(".", 1)[0] + ".jpg"

        original_url = upload_to_s3(
            raw, filename, mime_type or "image/jpeg", celebrant, photographer
        )
        compressed_url = upload_to_s3(
            compressed, out_name, "image/jpeg", celebrant, photographer
        )

        img = WeddingImage(
            filename=out_name,
            file_path=original_url,
            compressed_file_path=compressed_url,
            file_hash=file_hash,
            processed="processing",
            celebration_id=uuid.UUID(celebration_id),
        )
        db.add(img)
        db.commit()
        db.refresh(img)

        arr = load_image_from_bytes(compressed)
        faces = face_service.detect_and_encode_faces(arr)
        for f in faces:
            db.add(
                FaceVector(
                    image_id=img.id,
                    face_index=f["face_index"],
                    vector=f["vector"],
                    vector_pg=f["vector"],
                    bbox=f["bbox"],
                    landmarks=f["landmarks"],
                    confidence=f["confidence"],
                    quality_score=f["quality_score"],
                    celebration_id=img.celebration_id,
                    embedding_model=settings.EMBEDDING_MODEL_VERSION,
                )
            )

        img.faces_count = len(faces)
        img.processed = "completed"
        db.commit()

        logger.info(f"✅ Imported {out_name} ({len(faces)} faces)")
        _progress_incr(celebration_id)

    except Exception as e:
        logger.exception(f"❌ Drive import failed for {filename}: {e}")
        db.rollback()
        _progress_incr(celebration_id, failed=True)
    finally:
        db.close()
