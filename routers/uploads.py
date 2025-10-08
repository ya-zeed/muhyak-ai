import uuid
import json
import logging
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Depends,
    Form,
)
from sqlalchemy.orm import Session
from db import get_db, SessionLocal
from models import WeddingImage, FaceVector, Celebration
from schemas import ImageUploadResponse
from utils import (
    load_image_from_bytes,
    compress_image_bytes,
    calculate_file_hash,
)
from services import face_service, upload_to_s3, redis_client

router = APIRouter(prefix="/upload", tags=["upload"])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@router.post("", response_model=dict)
async def upload_wedding_photos(
    background_tasks: BackgroundTasks,
    celebrant: str = Form(...),
    photographer: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """
    ✅ Fast, Non-blocking Upload Endpoint
    - Accepts images immediately
    - Returns quickly (202 Accepted style)
    - Handles actual uploads + face processing in the background
    """
    # Validate celebration
    celebration = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer,
    ).first()

    if not celebration:
        raise HTTPException(404, "Celebration not found")

    filenames = [f.filename for f in files]
    logger.info(f"Received {len(files)} files from {photographer} for {celebrant}")

    # Read file contents now (before response)
    file_contents = [await f.read() for f in files]

    # Schedule background upload and processing
    background_tasks.add_task(
        _handle_background_uploads,
        celebrant,
        photographer,
        file_contents,
        filenames,
        celebration.id,
    )

    return {
        "status": "accepted",
        "count": len(files),
        "files": filenames,
        "message": "Images accepted and will be uploaded & processed in background.",
    }


def _handle_background_uploads(
    celebrant: str,
    photographer: str,
    contents: list[bytes],
    filenames: list[str],
    celebration_id: str,
):
    """
    🧠 Runs after FastAPI responds.
    Handles S3 uploads + DB insertion + face detection.
    """
    db = SessionLocal()
    for content, filename in zip(contents, filenames):
        try:
            if not content:
                logger.warning(f"Empty content for {filename}")
                continue

            file_hash = calculate_file_hash(content)

            existing = db.query(WeddingImage).filter(
                WeddingImage.file_hash == file_hash
            ).first()
            if existing:
                logger.info(f"Skipped duplicate file {filename}")
                continue

            # Upload to S3 (original + compressed)
            orig_url = upload_to_s3(
                content, filename, "image/jpeg", celebrant, photographer
            )
            comp_bytes = compress_image_bytes(content)
            comp_url = upload_to_s3(
                comp_bytes,
                f"compressed_{uuid.uuid4()}.jpg",
                "image/jpeg",
                celebrant,
                photographer,
            )

            # Create DB entry
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

            logger.info(f"Queued image {filename} for face processing.")

            # Face detection (still background)
            _process_image_faces(db, img, content)

        except Exception as e:
            logger.exception(f"Failed to handle {filename}: {e}")
            db.rollback()
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

        redis_client.setex(
            f"image_faces:{img.id}", 3600, json.dumps(faces, default=str)
        )

        logger.info(f"Processed {len(faces)} faces for {img.filename}")

    except Exception as e:
        logger.exception(f"Error processing image {img.filename}: {e}")
        img.processed = "failed"
        db.commit()
