"""Face search endpoints.

Uses pgvector's `<=>` cosine-distance operator with an HNSW index
on `face_vectors.vector_pg` for sub-100ms nearest-neighbor lookup.

Cosine similarity = 1 - cosine distance (pgvector's `<=>`).
"""

from __future__ import annotations

import logging
import uuid as uuid_module

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Path, UploadFile
from sqlalchemy.orm import Session

from db import get_db
from models import Celebration, FaceVector, WeddingImage
from schemas import FaceInfo, FaceSearchRequest, FaceSearchResponse
from services import face_service
from utils import load_image_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/{photographer}/{celebrant}/search", tags=["search"])


# Over-fetch factor: ask the index for more candidates than max_results
# so the per-image dedup + quality filter still has room to deliver
# `max_results` distinct images.
_OVERFETCH_MULT = 6
_OVERFETCH_FLOOR = 60

# Minimum per-face quality_score for a vector to be eligible as a search hit.
# Sub-0.2 vectors come from tiny / very blurry / off-axis faces and
# poison the similarity ranking with noise.
_MIN_QUALITY_FOR_SEARCH = 0.20

# MMR (Maximal Marginal Relevance) lambda: higher = prioritize relevance,
# lower = prioritize diversity. 0.7 picks similar people but spreads pose/scene.
_MMR_LAMBDA = 0.7


def _embed_vector(fv: FaceVector) -> np.ndarray:
    v = fv.vector_pg if fv.vector_pg is not None else fv.vector
    return np.asarray(v, dtype=np.float32)


def _mmr_rerank(
    hits: list[tuple[FaceVector, float]],
    max_results: int,
    threshold: float,
    lambda_param: float = _MMR_LAMBDA,
) -> list[tuple[FaceVector, float]]:
    """Pose-diversity reranking, dedup'd per image.

    `hits` is already sorted by descending similarity. For each pick we maximize
    ``lambda * sim(query, c) - (1-lambda) * max_sim(c, selected)`` so two
    near-identical shots of the same person at the same instant don't both
    survive. Faces below ``threshold`` are skipped entirely.
    InsightFace embeddings are L2-normalized, so dot product == cosine.
    """
    if not hits:
        return []

    selected: list[tuple[FaceVector, float]] = []
    selected_vecs: list[np.ndarray] = []
    seen_images: set = set()
    remaining = [(fv, sim) for fv, sim in hits if sim >= threshold]

    while remaining and len(selected) < max_results:
        if not selected_vecs:
            best_idx = 0
        else:
            best_idx = -1
            best_score = -float("inf")
            for i, (fv, sim) in enumerate(remaining):
                if fv.image_id in seen_images:
                    continue
                vec = _embed_vector(fv)
                penalty = max(float(np.dot(vec, s)) for s in selected_vecs)
                mmr = lambda_param * sim - (1.0 - lambda_param) * penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            if best_idx == -1:
                break

        fv, sim = remaining.pop(best_idx)
        if fv.image_id in seen_images:
            continue
        selected.append((fv, sim))
        selected_vecs.append(_embed_vector(fv))
        seen_images.add(fv.image_id)

    return selected


def _resolve_celebration(db: Session, photographer: str, celebrant: str) -> Celebration:
    celebration = (
        db.query(Celebration)
        .filter(Celebration.celebrant == celebrant, Celebration.photographer == photographer)
        .first()
    )
    if not celebration:
        logger.warning(f"Celebration not found: {photographer}/{celebrant}")
        raise HTTPException(404, "Celebration not found")
    return celebration


def _knn_search(db: Session, celebration_id, query_vector: list[float], k: int) -> list[tuple[FaceVector, float]]:
    """Run the indexed cosine-distance lookup. Returns (FaceVector, similarity)."""
    # pgvector requires the query to be a list/np-array of floats.
    distance_expr = FaceVector.vector_pg.cosine_distance(query_vector)
    rows = (
        db.query(FaceVector, distance_expr.label("distance"))
        .join(WeddingImage, FaceVector.image_id == WeddingImage.id)
        .filter(WeddingImage.celebration_id == celebration_id)
        .filter(WeddingImage.processed == "completed")
        .filter(FaceVector.vector_pg.isnot(None))
        .filter(
            (FaceVector.quality_score.is_(None))
            | (FaceVector.quality_score >= _MIN_QUALITY_FOR_SEARCH)
        )
        .order_by("distance")
        .limit(k)
        .all()
    )
    return [(fv, 1.0 - float(dist)) for fv, dist in rows]


def _build_response(db: Session, ranked: list[tuple[FaceVector, float]]) -> list[FaceSearchResponse]:
    """Hydrate ranked (FaceVector, similarity) tuples into API responses.

    `ranked` is expected to already be MMR'd: distinct per image, threshold-filtered.
    """
    out: list[FaceSearchResponse] = []

    for fv, similarity in ranked:
        img = db.get(WeddingImage, fv.image_id)
        if img is None:
            continue

        all_faces_in_image = db.query(FaceVector).filter(FaceVector.image_id == fv.image_id).all()
        all_faces = [
            FaceInfo(face_id=str(f.id), face_index=f.face_index, bbox=f.bbox)
            for f in all_faces_in_image
        ]

        out.append(
            FaceSearchResponse(
                image_id=str(fv.image_id),
                face_id=str(fv.id),
                filename=img.filename,
                similarity=similarity,
                face_index=fv.face_index,
                bbox=fv.bbox,
                file_path=img.file_path,
                compressed_file_path=img.compressed_file_path,
                compressed_url=img.compressed_file_path,
                thumbnail_url=img.compressed_file_path,
                all_faces=all_faces,
            )
        )

    return out


@router.post("/by-face/{face_id}", response_model=list[FaceSearchResponse])
async def search_by_face_id(
    photographer: str = Path(...),
    celebrant: str = Path(...),
    face_id: str = Path(..., description="UUID of the source face to search for"),
    request: FaceSearchRequest = Depends(),
    db: Session = Depends(get_db),
):
    """Search for similar faces using an existing face_id."""
    logger.info(f"🔍 Search by face_id: {photographer}/{celebrant}/{face_id}")

    try:
        face_uuid = uuid_module.UUID(face_id)
    except ValueError:
        raise HTTPException(400, "Invalid face_id format")

    source_face = db.query(FaceVector).filter(FaceVector.id == face_uuid).first()
    if not source_face:
        raise HTTPException(404, "Face not found")

    celebration = _resolve_celebration(db, photographer, celebrant)

    # Prefer the indexed pgvector column. Fall back to the legacy float[] only
    # if the source face hasn't been backfilled yet (shouldn't happen after migration 003).
    query_vec = source_face.vector_pg if source_face.vector_pg is not None else source_face.vector
    if query_vec is None:
        raise HTTPException(500, "Source face has no embedding")

    k = max(request.max_results * _OVERFETCH_MULT, _OVERFETCH_FLOOR)
    hits = _knn_search(db, celebration.id, list(query_vec), k=k)
    logger.info(f"📊 KNN returned {len(hits)} candidates (k={k})")
    ranked = _mmr_rerank(hits, max_results=request.max_results, threshold=request.threshold)
    logger.info(f"🎯 MMR returned {len(ranked)} ranked results")

    return _build_response(db, ranked)


@router.post("", response_model=list[FaceSearchResponse])
async def search_faces(
    photographer: str = Path(...),
    celebrant: str = Path(...),
    file: UploadFile = File(...),
    request: FaceSearchRequest = Depends(),
    db: Session = Depends(get_db),
):
    logger.info(f"🔍 Search: {photographer}/{celebrant} threshold={request.threshold}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    arr = load_image_from_bytes(await file.read())
    faces = face_service.detect_and_encode_faces(arr)
    if not faces:
        raise HTTPException(400, "No faces detected in search image")

    logger.info(f"✅ Detected {len(faces)} faces in search image")
    best = max(faces, key=lambda x: x["quality_score"])

    celebration = _resolve_celebration(db, photographer, celebrant)

    query_vec = best["vector"]
    # Sanity-log the embedding so we can spot model-mismatch bugs.
    q = np.asarray(query_vec, dtype=np.float32)
    logger.info(f"🔬 Query vec: dim={q.shape[0]}, norm={np.linalg.norm(q):.3f}")

    k = max(request.max_results * _OVERFETCH_MULT, _OVERFETCH_FLOOR)
    hits = _knn_search(db, celebration.id, query_vec, k=k)
    logger.info(f"📊 KNN returned {len(hits)} candidates (k={k})")
    if hits:
        logger.info(f"📈 Similarity range: min={hits[-1][1]:.3f} max={hits[0][1]:.3f}")
    ranked = _mmr_rerank(hits, max_results=request.max_results, threshold=request.threshold)
    logger.info(f"🎯 MMR returned {len(ranked)} ranked results")

    return _build_response(db, ranked)
