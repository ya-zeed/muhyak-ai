from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Depends, Path
from sqlalchemy.orm import Session, selectinload

from db import get_db
from models import WeddingImage, FaceVector, Celebration

router = APIRouter(prefix="/{photographer}/{celebrant}/images", tags=["images"])

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
    
    q = q.options(selectinload(WeddingImage.faces)).order_by(
        WeddingImage.order_number.is_(None),
        WeddingImage.order_number.asc().nulls_last(),
        WeddingImage.upload_date.asc()
    )
    
    imgs = (
        q.options(selectinload(WeddingImage.faces))
        .offset(skip)
        .limit(limit)
        .all()
    )

    total = db.query(WeddingImage).filter(WeddingImage.celebration_id == celebration.id).count()

    return {
        "data": [
        {
            "image_id": str(img.id),
            "filename": img.filename,
            "upload_date": img.upload_date.isoformat(),
            "faces_count": img.faces_count,
            "processed": img.processed,
            "high_quality_url": img.file_path,
            "compressed_url": img.compressed_file_path,
            "order_number": img.order_number,
            "faces": [
                {
                    "face_id": str(face.id),
                    "face_index": face.face_index,
                    "bbox": face.bbox,
                    "landmarks": face.landmarks,
                    "confidence": face.confidence,
                    "quality_score": face.quality_score,
                    "created_date": face.created_date.isoformat(),
                }
                for face in img.faces
            ]
        }
        for img in imgs
    ],
        "total": total
    }
    
    
@router.get("/by-face/{face_id}")
def get_image_by_face(face_id: str, db: Session = Depends(get_db)):
    face = db.query(FaceVector).options(selectinload(FaceVector.image)).filter(FaceVector.id == face_id).first()
    if not face:
        raise HTTPException(404, "Face not found")
    img = face.image
    return {
        "image_id": str(img.id),
        "filename": img.filename,
        "upload_date": img.upload_date.isoformat(),
        "faces_count": img.faces_count,
        "processed": img.processed,
        "high_quality_url": img.file_path,
        "compressed_url": img.compressed_file_path,
        "faces": [
            {
                "face_id": str(f.id),
                "face_index": f.face_index,
                "bbox": f.bbox,
                "landmarks": f.landmarks,
                "confidence": f.confidence,
                "quality_score": f.quality_score,
                "created_date": f.created_date.isoformat(),
            }
            for f in img.faces
        ]
    }
    
@router.get("/{image_id}")
def get_image(image_id: str, db: Session = Depends(get_db)):
    img = db.query(WeddingImage).options(selectinload(WeddingImage.faces)).filter(WeddingImage.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    return {
        "image_id": str(img.id),
        "filename": img.filename,
        "upload_date": img.upload_date.isoformat(),
        "faces_count": img.faces_count,
        "processed": img.processed,
        "high_quality_url": img.file_path,
        "compressed_url": img.compressed_file_path,
        "faces": [
            {
                "face_id": str(f.id),
                "face_index": f.face_index,
                "bbox": f.bbox,
                "landmarks": f.landmarks,
                "confidence": f.confidence,
                "quality_score": f.quality_score,
                "created_date": f.created_date.isoformat(),
            }
            for f in img.faces
        ]
    }
    
    
@router.patch("/{image_id}/order")
def update_image_order(
    photographer: str,
    celebrant: str,
    image_id: str = Path(...),
    order_number: int = Body(..., embed=True),
    db: Session = Depends(get_db),
):
    celebration = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer
    ).first()
    if not celebration:
        raise HTTPException(404, "Celebration not found")

    image = (
        db.query(WeddingImage)
        .filter(
            WeddingImage.id == image_id,
            WeddingImage.celebration_id == celebration.id
        )
        .first()
    )
    if not image:
        raise HTTPException(404, "Image not found")

    image.order_number = order_number
    db.commit()
    db.refresh(image)

    return {
        "message": "Image order updated successfully",
        "image_id": str(image.id),
        "order_number": image.order_number
    }


@router.delete("/{image_id}")
def delete_image(image_id: str, db: Session = Depends(get_db)):
    img = db.get(WeddingImage, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    db.query(FaceVector).filter(FaceVector.image_id == image_id).delete()
    db.delete(img)
    db.commit()
    return {"message": "Image deleted successfully"}
