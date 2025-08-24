import uuid
from urllib.parse import urlparse
import boto3
import insightface
import redis
from typing import Any, Dict, List, Tuple
import cv2

from config import settings

_s3 = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
    endpoint_url=settings.S3_ENDPOINT,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)


def upload_to_s3(file_bytes: bytes, filename: str, content_type: str, celebrant: str, photographer: str) -> str:
    if not settings.AWS_S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET not configured")
    key = f"{photographer}/{celebrant}/{uuid.uuid4()}_{filename}"
    _s3.put_object(
        Bucket=settings.AWS_S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
        ACL="public-read"
    )

    # Prefer explicit public base if provided (CDN/custom domain)
    if settings.PUBLIC_S3_BASE_URL:
        return f"{settings.PUBLIC_S3_BASE_URL}/{key}"

    # If using a Spaces endpoint, build virtual-hosted URL: bucket.endpoint/key
    if settings.S3_ENDPOINT:
        host = urlparse(settings.S3_ENDPOINT).netloc  # e.g., nyc3.digitaloceanspaces.com
        return f"https://{settings.AWS_S3_BUCKET}.{host}/{key}"

    # Fallback: standard AWS S3 format
    return f"https://{settings.AWS_S3_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"


redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


class FaceRecognitionService:
    def __init__(self):
        self._app = None
        self._init_models()

    def _init_models(self):
        try:
            self._app = insightface.app.FaceAnalysis(providers=[settings.INSIGHTFACE_PROVIDER])
            self._app.prepare(ctx_id=0, det_size=(settings.DET_SIZE_W, settings.DET_SIZE_H))
        except Exception as e:
            raise

    def detect_and_encode_faces(self, image_bgr) -> List[Dict[str, Any]]:
        faces = self._app.get(image_bgr)
        out: List[Dict[str, Any]] = []
        for i, f in enumerate(faces):
            emb = f.embedding
            if emb is None or emb.shape[0] != settings.VECTOR_DIM:
                # if model mismatch, skip to keep vectors consistent
                continue
            out.append({
                "face_index": i,
                "vector": emb.tolist(),
                "bbox": f.bbox.tolist(),
                "landmarks": f.kps.flatten().tolist(),
                "confidence": float(f.det_score),
                "quality_score": self._quality_score(f, image_bgr),
            })
        return out

    def _quality_score(self, face, image_bgr) -> float:
        x1, y1, x2, y2 = face.bbox.astype(int)
        crop = image_bgr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp = min(sharpness / 1000, 1.0)
        area = max((y2 - y1) * (x2 - x1), 1)
        size = min(area / (100 * 100), 1.0)
        conf = float(min(face.det_score, 1.0))
        return float(sharp * 0.4 + size * 0.3 + conf * 0.3)


face_service = FaceRecognitionService()
