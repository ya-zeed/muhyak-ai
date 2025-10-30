import os
import uuid
import json
import logging
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Form,
)
from sqlalchemy.orm import Session
from db import get_db, SessionLocal
from models import WeddingImage, FaceVector, Celebration
from utils import (
    load_image_from_bytes,
    compress_image_bytes,
    calculate_file_hash,
)
from services import face_service, upload_to_s3, redis_client
import redis
from rq import Queue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/upload", tags=["upload"])

# 🔌 RQ setup from environment
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = redis.from_url(redis_url)
queue = Queue("default", connection=redis_conn)


@router.post("", response_model=dict)
async def upload_wedding_photos(
    celebrant: str = Form(...),
    photographer: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    ✅ Upload Endpoint (now queues background jobs)
    - Accepts multiple images quickly
    - Immediately returns (non-blocking)
    - Each image is processed by a separate Redis RQ job
    """
    celebration = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer,
    ).first()

    if not celebration:
        raise HTTPException(404, "Celebration not found")

    filenames = [f.filename for f in files]
    logger.info(f"📸 Received {len(files)} files from {photographer} for {celebrant}")

    # Read file contents before responding
    file_contents = [await f.read() for f in files]

    # Queue each file for async processing
    for content, filename in zip(file_contents, filenames):
        queue.enqueue(
            _handle_single_upload,
            celebrant,
            photographer,
            filename,
            content,
            celebration.id,
        )

    return {
        "status": "accepted",
        "count": len(files),
        "files": filenames,
        "message": "Images accepted and queued for background processing.",
    }


# --------------------------
# 🔧 Background Job Functions
# --------------------------

def _handle_single_upload(
    celebrant: str,
    photographer: str,
    filename: str,
    content: bytes,
    celebration_id: str,
):
    """
    🧠 Runs inside the RQ worker.
    Handles S3 uploads + DB insert + face detection.
    """
    db = SessionLocal()

    try:
        if not content:
            logger.warning(f"⚠️ Empty content for {filename}")
            return

        file_hash = calculate_file_hash(content)

        existing = db.query(WeddingImage).filter(
            WeddingImage.file_hash == file_hash
        ).first()
        if existing:
            logger.info(f"🟡 Skipped duplicate file {filename}")
            return

        # Upload to S3 (original + compressed)
        orig_url = upload_to_s3(content, filename, "image/jpeg", celebrant, photographer)
        comp_bytes = compress_image_bytes(content)
        comp_url = upload_to_s3(
            comp_bytes,
            f"compressed_{uuid.uuid4()}.jpg",
            "image/jpeg",
            celebrant,
            photographer,
        )

        # DB entry
        img = WeddingImage(
            filename=filename,
            file_path=orig_url,
            compressed_file_path=comp_url,
            file_hash=file_hash,
            processed="pending",
            celebration_id=celebration_id,
        )
        db.add(img)
        db.commit()
        db.refresh(img)

        logger.info(f"🧾 Added {filename}, starting face detection...")

        _process_image_faces(db, img, content)

    except Exception as e:
        logger.exception(f"❌ Failed to handle {filename}: {e}")
        db.rollback()
    finally:
        db.close()


def _process_image_faces(db: Session, img: WeddingImage, file_content: bytes):
    """
    🔍 Detects faces & saves vectors.
    """
    try:
        img.processed = "processing"
        db.commit()

        arr = load_image_from_bytes(file_content)
        faces = face_service.detect_and_encode_faces(arr)

        for f in faces:
            db.add(
                FaceVector(
                    image_id=img.id,
                    face_index=f["face_index"],
                    vector=f["vector"],
                    bbox=f["bbox"],
                    landmarks=f["landmarks"],
                    confidence=f["confidence"],
                    quality_score=f["quality_score"],
                    celebration_id=img.celebration_id,
                )
            )

        img.faces_count = len(faces)
        img.processed = "completed"
        db.commit()

        redis_client.setex(f"image_faces:{img.id}", 3600, json.dumps(faces, default=str))

        logger.info(f"✅ Processed {len(faces)} faces for {img.filename}")

    except Exception as e:
        logger.exception(f"💥 Error processing {img.filename}: {e}")
        img.processed = "failed"
        db.commit()
