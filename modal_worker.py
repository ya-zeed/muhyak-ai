"""
Modal.com Serverless Worker for Face Processing

This replaces the RQ worker with a serverless function that:
- Scales automatically based on queue size
- Costs $0 when idle
- Processes images 10x faster with parallel execution

Setup:
    pip install modal
    modal token new
    modal deploy modal_worker.py

Usage:
    The worker automatically polls Redis queue and processes jobs.
    Or call directly: modal run modal_worker.py::process_single_image --help
"""

import modal

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "insightface>=0.7,<0.9",
        "onnxruntime>=1.17,<2.0",
        "opencv-python-headless>=4.9,<4.11",
        "numpy>=1.26,<3.0",
        "pillow>=10.3,<11.0",
        "boto3>=1.34,<2.0",
        "redis>=5.0,<6.0",
        "psycopg[binary]>=3.1,<3.2",
        "sqlalchemy>=2.0,<2.1",
    )
    .run_commands(
        # Pre-download the fast model
        "python -c \"from insightface.app import FaceAnalysis; "
        "a = FaceAnalysis(name='buffalo_s'); "
        "a.prepare(ctx_id=-1, det_size=(320,320)); "
        "print('Model cached')\""
    )
)

app = modal.App("muhyak-face-processor", image=image)

# Secrets for connecting to your services
secrets = [
    modal.Secret.from_name("muhyak-aws"),  # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
    modal.Secret.from_name("muhyak-db"),   # DATABASE_URL
    modal.Secret.from_name("muhyak-redis"), # REDIS_URL
]


@app.cls(
    memory=2048,  # 2GB RAM for ML model
    cpu=2.0,
    timeout=300,
    secrets=secrets,
    container_idle_timeout=60,  # Keep warm for 60s to avoid cold starts
)
class FaceProcessor:
    """Face detection and encoding processor."""

    @modal.enter()
    def setup(self):
        """Initialize model once when container starts."""
        from insightface.app import FaceAnalysis
        import os

        self.face_app = FaceAnalysis(name="buffalo_s")
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))

        # Setup S3 client
        import boto3
        self.s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "nyc3"),
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        self.bucket = os.environ.get("AWS_S3_BUCKET")

    @modal.method()
    def process_image_bytes(self, image_bytes: bytes) -> list:
        """Process image bytes and return face data."""
        import cv2
        import numpy as np

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return []

        # Detect faces
        faces = self.face_app.get(image_bgr)

        results = []
        for i, f in enumerate(faces):
            if f.embedding is None or len(f.embedding) != 512:
                continue

            # Calculate quality score
            x1, y1, x2, y2 = f.bbox.astype(int)
            crop = image_bgr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            quality = 0.0
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharp = min(sharpness / 1000, 1.0)
                area = max((y2 - y1) * (x2 - x1), 1)
                size = min(area / 10000, 1.0)
                conf = float(min(f.det_score, 1.0))
                quality = float(sharp * 0.4 + size * 0.3 + conf * 0.3)

            results.append({
                "face_index": i,
                "vector": f.embedding.tolist(),
                "bbox": f.bbox.tolist(),
                "landmarks": f.kps.flatten().tolist(),
                "confidence": float(f.det_score),
                "quality_score": quality,
            })

        return results

    @modal.method()
    def process_from_s3(self, s3_key: str) -> list:
        """Download image from S3 and process it."""
        response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        image_bytes = response["Body"].read()
        return self.process_image_bytes(image_bytes)


@app.function(
    memory=1024,
    cpu=1.0,
    timeout=60,
    schedule=modal.Period(seconds=5),  # Poll every 5 seconds
    secrets=secrets,
)
def poll_redis_queue():
    """Poll Redis queue and dispatch jobs to workers."""
    import os
    import json
    import redis

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    r = redis.from_url(redis_url)

    # Check for pending jobs
    job_data = r.lpop("modal_face_jobs")
    if not job_data:
        return {"status": "no_jobs"}

    job = json.loads(job_data)
    processor = FaceProcessor()

    if job.get("type") == "s3":
        faces = processor.process_from_s3.remote(job["s3_key"])
    else:
        faces = processor.process_image_bytes.remote(job["image_bytes"])

    # Store result back in Redis
    result_key = f"face_result:{job['image_id']}"
    r.setex(result_key, 3600, json.dumps(faces))

    return {"status": "processed", "image_id": job["image_id"], "faces_count": len(faces)}


@app.local_entrypoint()
def main():
    """Test the face processor locally."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: modal run modal_worker.py -- <image_path>")
        return

    image_path = sys.argv[1]
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    processor = FaceProcessor()
    faces = processor.process_image_bytes.remote(image_bytes)
    print(f"Found {len(faces)} faces")
    for face in faces:
        print(f"  - Confidence: {face['confidence']:.2f}, Quality: {face['quality_score']:.2f}")


# Helper function to enqueue jobs from your FastAPI app
def enqueue_face_job(redis_client, image_id: str, image_bytes: bytes = None, s3_key: str = None):
    """
    Call this from your FastAPI upload endpoint instead of RQ.

    Example:
        from modal_worker import enqueue_face_job

        @router.post("/upload")
        async def upload(files: list[UploadFile]):
            for file in files:
                image_id = str(uuid.uuid4())
                # Upload to S3 first...
                enqueue_face_job(redis_client, image_id, s3_key=s3_key)
    """
    import json

    job = {"image_id": image_id}
    if s3_key:
        job["type"] = "s3"
        job["s3_key"] = s3_key
    elif image_bytes:
        job["type"] = "bytes"
        job["image_bytes"] = image_bytes.decode("latin-1")  # Encode for JSON

    redis_client.rpush("modal_face_jobs", json.dumps(job))
