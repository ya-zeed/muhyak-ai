from pydantic import BaseModel
from typing import Any, List, Optional, Dict

class FaceSearchRequest(BaseModel):
    threshold: float = 0.6
    max_results: int = 50

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

class FaceClusterResponse(BaseModel):
    cluster_id: int
    face_count: int
    representative_face: Dict[str, Any]
    faces: List[Dict[str, Any]]
