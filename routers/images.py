from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from db import get_db
from models import WeddingImage, FaceVector, Celebration

router = APIRouter(prefix="/images", tags=["images"])

@router.get("")
def list_images(skip: int = 0, limit: int = 100, status: str | None = None, celebrant: str = "", photographer: str = "", db: Session = Depends(get_db)):
    celebration = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer
    ).first()

    if not celebration:
        raise HTTPException(404, "Celebration not found")

    q = db.query(WeddingImage)
    if status:
        q = q.filter(WeddingImage.processed == status)

    q = q.filter(WeddingImage.celebration_id == celebration.id)
    imgs = q.offset(skip).limit(limit).all()
    return [
        {
            "image_id": str(img.id),
            "filename": img.filename,
            "upload_date": img.upload_date.isoformat(),
            "faces_count": img.faces_count,
            "processed": img.processed,
            "high_quality_url": img.file_path,
            "compressed_url": img.compressed_file_path,
        }
        for img in imgs
    ]

@router.delete("/{image_id}")
def delete_image(image_id: str, db: Session = Depends(get_db)):
    img = db.get(WeddingImage, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    db.query(FaceVector).filter(FaceVector.image_id == image_id).delete()
    db.delete(img)
    db.commit()
    return {"message": "Image deleted successfully"}
