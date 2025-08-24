import uuid, json

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form
from sqlalchemy.orm import Session

from db import get_db
from models import WeddingImage, FaceVector, Celebration
from schemas import ImageUploadResponse
from utils import load_image_from_bytes, compress_image_bytes, calculate_file_hash
from services import face_service, upload_to_s3, redis_client

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=list[ImageUploadResponse])
async def upload_wedding_photos(
        background_tasks: BackgroundTasks,
        celebrant: str = Form(...),
        photographer: str = Form(...),
        files: list[UploadFile] = File(...),
        db: Session = Depends(get_db),
):
    uploaded: list[ImageUploadResponse] = []
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, f"{file.filename} is not an image")

        content = await file.read()
        file_hash = calculate_file_hash(content)

        existing = db.query(WeddingImage).filter(WeddingImage.file_hash == file_hash).first()
        if existing:
            uploaded.append(ImageUploadResponse(
                image_id=str(existing.id), filename=existing.filename,
                faces_detected=existing.faces_count, status="already_exists",
                compressed_url=existing.compressed_file_path
            ))
            continue

        # get celebration by photographer and celebrant
        celebration = db.query(Celebration).filter(
            Celebration.celebrant == celebrant,
            Celebration.photographer == photographer
        ).first()

        if not celebration:
            raise HTTPException(404, "Celebration not found")

        celebration_id = celebration.id

        # store in S3
        orig_url = upload_to_s3(content, file.filename, file.content_type, celebrant, photographer)
        comp_bytes = compress_image_bytes(content)
        comp_url = upload_to_s3(comp_bytes, f"compressed_{uuid.uuid4()}.jpg", "image/jpeg", celebrant, photographer)

        img = WeddingImage(
            filename=file.filename, file_path=orig_url,
            compressed_file_path=comp_url, file_hash=file_hash, processed="pending", celebration_id=celebration_id,
        )
        db.add(img)
        db.commit()
        db.refresh(img)

        background_tasks.add_task(_process_image_faces, str(img.id), content)
        uploaded.append(ImageUploadResponse(
            image_id=str(img.id), filename=file.filename,
            faces_detected=0, status="processing", compressed_url=comp_url
        ))
    return uploaded


def _process_image_faces(image_id: str, file_content: bytes):
    from db import SessionLocal
    db = SessionLocal()
    try:
        img = db.get(WeddingImage, image_id)
        if not img: return
        img.processed = "processing"
        db.commit()

        arr = load_image_from_bytes(file_content)
        faces = face_service.detect_and_encode_faces(arr)

        for f in faces:
            db.add(FaceVector(
                image_id=img.id,
                face_index=f["face_index"],
                vector=f["vector"],
                bbox=f["bbox"],
                landmarks=f["landmarks"],
                confidence=f["confidence"],
                quality_score=f["quality_score"],
                celebration_id=img.celebration_id,
            ))

        img.faces_count = len(faces)
        img.processed = "completed"
        db.commit()
        redis_client.setex(f"image_faces:{image_id}", 3600, json.dumps(faces))
    except Exception as e:
        if img := db.get(WeddingImage, image_id):
            img.processed = "failed"
            db.commit()
        print(f"Error processing image {image_id}: {e}")
    finally:
        db.close()
