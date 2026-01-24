
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Path
from sqlalchemy.orm import Session

from db import get_db

logger = logging.getLogger(__name__)
from models import FaceVector, WeddingImage, Celebration
from schemas import FaceSearchRequest, FaceSearchResponse, FaceInfo
from utils import load_image_from_bytes, cosine_similarity_search
from services import face_service
from sqlalchemy import and_


router = APIRouter(prefix="/{photographer}/{celebrant}/search", tags=["search"])


@router.post("/by-face/{face_id}", response_model=list[FaceSearchResponse])
async def search_by_face_id(
        photographer: str = Path(...),
        celebrant: str = Path(...),
        face_id: str = Path(..., description="UUID of the source face to search for"),
        request: FaceSearchRequest = Depends(),
        db: Session = Depends(get_db),
):
    """Search for similar faces using an existing face_id."""
    import uuid as uuid_module
    logger.info(f"🔍 Search by face_id: photographer={photographer}, celebrant={celebrant}, face_id={face_id}")

    # Validate and parse face_id
    try:
        face_uuid = uuid_module.UUID(face_id)
    except ValueError:
        raise HTTPException(400, "Invalid face_id format")

    # Get source face
    source_face = db.query(FaceVector).filter(FaceVector.id == face_uuid).first()
    if not source_face:
        logger.warning(f"Face not found: {face_id}")
        raise HTTPException(404, "Face not found")

    # Get celebration
    celebration = db.query(Celebration).filter(
        Celebration.celebrant == celebrant,
        Celebration.photographer == photographer
    ).first()
    if not celebration:
        logger.warning(f"Celebration not found: {photographer}/{celebrant}")
        raise HTTPException(404, "Celebration not found")

    # Get all face vectors for this celebration (including source face's image)
    vectors = (
        db.query(FaceVector)
        .join(WeddingImage)
        .filter(WeddingImage.processed == "completed")
        .filter(WeddingImage.celebration_id == celebration.id)
        .all()
    )

    logger.info(f"📊 Found {len(vectors)} face vectors to compare against")

    if not vectors:
        return []

    cand = [fv.vector for fv in vectors]
    sims = cosine_similarity_search(source_face.vector, cand, threshold=request.threshold)
    logger.info(f"🎯 Found {len(sims)} matches above threshold {request.threshold}")

    results: list[FaceSearchResponse] = []
    seen_images = set()  # Track seen images to avoid duplicates

    for idx, score in sims[:request.max_results]:
        fv = vectors[idx]

        # Skip if we already have this image
        if fv.image_id in seen_images:
            continue
        seen_images.add(fv.image_id)

        img = db.get(WeddingImage, fv.image_id)

        # Get all faces for this image
        all_faces_in_image = db.query(FaceVector).filter(FaceVector.image_id == fv.image_id).all()
        all_faces = [
            FaceInfo(
                face_id=str(f.id),
                face_index=f.face_index,
                bbox=f.bbox
            )
            for f in all_faces_in_image
        ]

        results.append(FaceSearchResponse(
            image_id=str(fv.image_id),
            face_id=str(fv.id),
            filename=img.filename,
            similarity_score=score,
            face_index=fv.face_index,
            bbox=fv.bbox,
            image_url=img.file_path,
            compressed_url=img.compressed_file_path,
            all_faces=all_faces,
        ))
    return results


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

    # Debug: check vector dimensions and sample values
    import numpy as np
    query_vec = np.array(best["vector"])
    db_vec = np.array(cand[0])
    logger.info(f"🔬 Query vector: len={len(query_vec)}, norm={np.linalg.norm(query_vec):.3f}, sample={query_vec[:3]}")
    logger.info(f"🔬 DB vector[0]: len={len(db_vec)}, norm={np.linalg.norm(db_vec):.3f}, sample={db_vec[:3]}")

    # Debug: check similarity scores before filtering
    from sklearn.metrics.pairwise import cosine_similarity
    all_sims = cosine_similarity([best["vector"]], cand)[0]
    logger.info(f"📈 Similarity scores - min: {all_sims.min():.3f}, max: {all_sims.max():.3f}, mean: {all_sims.mean():.3f}")

    sims = cosine_similarity_search(best["vector"], cand, threshold=request.threshold)
    logger.info(f"🎯 Found {len(sims)} matches above threshold {request.threshold}")

    results: list[FaceSearchResponse] = []
    seen_images = set()  # Track seen images to avoid duplicates

    for idx, score in sims[:request.max_results]:
        fv = vectors[idx]

        # Skip if we already have this image
        if fv.image_id in seen_images:
            continue
        seen_images.add(fv.image_id)

        img = db.get(WeddingImage, fv.image_id)

        # Get all faces for this image
        all_faces_in_image = db.query(FaceVector).filter(FaceVector.image_id == fv.image_id).all()
        all_faces = [
            FaceInfo(
                face_id=str(f.id),
                face_index=f.face_index,
                bbox=f.bbox
            )
            for f in all_faces_in_image
        ]

        results.append(FaceSearchResponse(
            image_id=str(fv.image_id),
            face_id=str(fv.id),
            filename=img.filename,
            similarity_score=score,
            face_index=fv.face_index,
            bbox=fv.bbox,
            image_url=img.file_path,
            compressed_url=img.compressed_file_path,
            all_faces=all_faces,
        ))
    return results
