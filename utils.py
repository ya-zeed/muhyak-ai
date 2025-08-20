import io, hashlib
from typing import Tuple
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def calculate_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def compress_image_bytes(image_bytes: bytes, quality: int = 75, max_size: Tuple[int,int] = (1024,1024)) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size, resample=Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def cosine_similarity_search(query_vector, candidate_vectors, threshold: float = 0.6):
    sims = cosine_similarity([query_vector], candidate_vectors)[0]
    results = [(i, float(s)) for i, s in enumerate(sims) if s >= threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results
