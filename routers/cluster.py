from typing import Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db import get_db
from models import FaceVector, WeddingImage
from schemas import FaceClusterResponse

router = APIRouter(prefix="/cluster", tags=["cluster"])

@router.post("", response_model=list[FaceClusterResponse])
def cluster_faces(min_cluster_size: int = 2, eps: float = 0.4, db: Session = Depends(get_db)):
    vectors = db.query(FaceVector).join(WeddingImage).filter(WeddingImage.processed == "completed").all()
    if len(vectors) < min_cluster_size:
        return []

    data = np.array([fv.vector for fv in vectors], dtype=np.float32)
    clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="cosine").fit(data)
    labels = clustering.labels_

    clusters: Dict[int, list[tuple[FaceVector, int]]] = {}
    for i, lbl in enumerate(labels):
        if lbl == -1: continue
        clusters.setdefault(lbl, []).append((vectors[i], i))

    results: list[FaceClusterResponse] = []
    for cid, items in clusters.items():
        best_face, _ = max(items, key=lambda x: (x[0].quality_score or 0))
        rep_img = db.get(WeddingImage, best_face.image_id)
        faces_list = [
            {"image_id": str(fv.image_id), "filename": db.get(WeddingImage, fv.image_id).filename,
             "face_index": fv.face_index, "bbox": fv.bbox, "quality_score": fv.quality_score}
            for fv, _ in items
        ]
        results.append(FaceClusterResponse(
            cluster_id=cid,
            face_count=len(items),
            representative_face={
                "image_id": str(best_face.image_id), "filename": rep_img.filename,
                "face_index": best_face.face_index, "bbox": best_face.bbox,
                "quality_score": best_face.quality_score,
            },
            faces=faces_list
        ))
    return results
