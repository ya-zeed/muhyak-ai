
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Path
from sqlalchemy.orm import Session

from db import get_db

logger = logging.getLogger(__name__)
from models import FaceVector, WeddingImage, Celebration
from schemas import FaceSearchRequest, FaceSearchResponse
from utils import load_image_from_bytes, cosine_similarity_search
from services import face_service
from sqlalchemy import and_


router = APIRouter(prefix="/{photographer}/{celebrant}/search", tags=["search"])


@router.post("", response_model=list[FaceSearchResponse])
async def search_faces(
        photographer: str = Path(...),
        celebrant: str = Path(...),
        file: UploadFile = File(...),
        request: FaceSearchRequest = Depends(),
        db: Session = Depends(get_db),
):
    logger.info(f"🔍 Search request: photographer={photographer}, celebrant={celebrant}, threshold={request.threshold}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    arr = load_image_from_bytes(await file.read())
    faces = face_service.detect_and_encode_faces(arr)
    if not faces:
        logger.warning("No faces detected in search image")
        raise HTTPException(400, "No faces detected in search image")

    logger.info(f"✅ Detected {len(faces)} faces in search image")
    best = max(faces, key=lambda x: x["quality_score"])

    # Check if celebration exists
    celebration = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer
    ).first()
    logger.info(f"📋 Celebration found: {celebration.id if celebration else 'None'}")

    vectors = (db.query(FaceVector).join(WeddingImage).filter(WeddingImage.processed == "completed")
               .filter(
        WeddingImage.celebration.has(
            and_(
                Celebration.celebrant == celebrant,
                Celebration.photographer == photographer
            )
        )
    )
               .all())

    logger.info(f"📊 Found {len(vectors)} face vectors in database for this celebration")

    if not vectors:
        return []

    cand = [fv.vector for fv in vectors]
    sims = cosine_similarity_search(best["vector"], cand, threshold=request.threshold)
    logger.info(f"🎯 Found {len(sims)} matches above threshold {request.threshold}")

    results: list[FaceSearchResponse] = []
    for idx, score in sims[:request.max_results]:
        fv = vectors[idx]
        img = db.get(WeddingImage, fv.image_id)
        results.append(FaceSearchResponse(
            image_id=str(fv.image_id),
            filename=img.filename,
            similarity_score=score,
            face_index=fv.face_index,
            bbox=fv.bbox,
            image_url=img.file_path,
            compressed_url=img.compressed_file_path,
        ))
    return results
