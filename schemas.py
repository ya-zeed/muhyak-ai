from pydantic import BaseModel
from typing import Any, List, Optional, Dict

class FaceSearchRequest(BaseModel):
    threshold: float = 0.6
    max_results: int = 50

class FaceSearchResponse(BaseModel):
    image_id: str
    filename: str
    similarity_score: float
    face_index: int
    bbox: List[float]
    image_url: Optional[str]
    compressed_url: Optional[str]

class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    faces_detected: int
    status: str
    compressed_url: Optional[str]

class FaceClusterResponse(BaseModel):
    cluster_id: int
    face_count: int
    representative_face: Dict[str, Any]
    faces: List[Dict[str, Any]]
