from pydantic import BaseModel, field_validator
from typing import Any, List, Optional, Dict

# Server-side safety bounds. The threshold/max_results come from the client, so
# without a floor a guest could send threshold=0&max_results=100000 and pull
# back other people's (incl. women's) photos — defeating face-search privacy.
MIN_SEARCH_THRESHOLD = 0.45
MAX_SEARCH_RESULTS = 200

class FaceSearchRequest(BaseModel):
    threshold: float = 0.6
    max_results: int = 50

    @field_validator("threshold")
    @classmethod
    def _clamp_threshold(cls, v: float) -> float:
        # Never let the match threshold drop below the privacy floor.
        return max(MIN_SEARCH_THRESHOLD, min(float(v), 1.0))

    @field_validator("max_results")
    @classmethod
    def _clamp_max_results(cls, v: int) -> int:
        return max(1, min(int(v), MAX_SEARCH_RESULTS))

class FaceInfo(BaseModel):
    face_id: str
    face_index: int
    bbox: List[float]

class FaceSearchResponse(BaseModel):
    image_id: str
    face_id: str  # The matched face
    filename: str
    similarity: float
    face_index: int
    bbox: List[float]
    file_path: Optional[str]
    compressed_file_path: Optional[str]
    compressed_url: Optional[str]
    thumbnail_url: Optional[str]
    all_faces: List[FaceInfo] = []  # All faces in this image

class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    faces_detected: int
    status: str
    compressed_url: Optional[str]

