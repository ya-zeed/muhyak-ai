"""
Modal Serverless Worker for Muhyak AI

Provides serverless face processing and quality analysis functions.
Scales automatically, costs $0 when idle.

Setup:
    1. Install Modal: pip install modal
    2. Authenticate: modal token new
    3. Create secrets in Modal dashboard:
       - muhyak-env: DATABASE_URL, REDIS_URL, AWS_*, S3_ENDPOINT, etc.
    4. Deploy: modal deploy modal_worker.py

Usage:
    Jobs are automatically dispatched when WORKER_BACKEND=modal in .env
"""
import modal

# Face-model config — must match local muhyak-ai/config.py.
# After changing INSIGHTFACE_MODEL or DET_SIZE you must redeploy
# (modal deploy modal_worker.py) so the new model is pre-cached in the image.
INSIGHTFACE_MODEL = "buffalo_l"
DET_SIZE = 640
MIN_FACE_PIXELS = 48
EMBEDDING_MODEL_VERSION = "buffalo_l_v1"

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "insightface>=0.7,<0.9",
        "onnxruntime>=1.17,<2.0",
        "opencv-python-headless>=4.9,<4.11",
        "numpy>=1.26,<3.0",
        "pillow>=10.3,<11.0",
        "pillow-heif>=0.16,<1.0",
        "boto3>=1.34,<2.0",
        "redis>=5.0,<6.0",
        "psycopg[binary]>=3.1,<3.2",
        "sqlalchemy>=2.0,<2.1",
        "pgvector>=0.3,<0.5",
    )
    .run_commands(
        # Pre-download the face model into the image layer so cold starts don't pay for it.
        f"python -c \""
        f"from insightface.app import FaceAnalysis; "
        f"a = FaceAnalysis(name='{INSIGHTFACE_MODEL}'); "
        f"a.prepare(ctx_id=-1, det_size=({DET_SIZE},{DET_SIZE})); "
        f"print('Model {INSIGHTFACE_MODEL} cached')\""
    )
)

app = modal.App("muhyak-face-processor", image=image)

# Load secrets from Modal
secrets = [modal.Secret.from_name("muhyak")]


# Shared setup for database and S3
def get_db_session():
    """Create a database session."""
    import os
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()


def get_s3_client():
    """Create S3 client."""
    import os
    import boto3

    return boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "nyc3"),
        endpoint_url=os.environ.get("S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def get_redis_client():
    """Create Redis client."""
    import os
    import redis

    return redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=True)


def extract_s3_key(file_path: str) -> str:
    """
    Extract S3 key from various URL formats:
    - https://bucket.region.cdn.digitaloceanspaces.com/path/to/file.jpg
    - https://bucket.region.digitaloceanspaces.com/path/to/file.jpg
    - https://bucket.s3.region.amazonaws.com/path/to/file.jpg
    """
    from urllib.parse import urlparse

    parsed = urlparse(file_path)
    # Remove leading slash from path
    key = parsed.path.lstrip("/")
    # Remove query string if present
    key = key.split("?")[0]
    return key


def _decode_image_with_exif(image_bytes: bytes):
    """Decode bytes -> BGR ndarray, honoring EXIF rotation and HEIC.

    Returns None when the bytes can't be decoded so callers can mark the image failed.
    """
    import io
    import logging
    import numpy as np
    import cv2
    from PIL import Image, ImageOps

    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.getLogger(__name__).warning(f"PIL decode failed ({e}); falling back to cv2")
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.function(
    memory=2048,
    cpu=2.0,
    timeout=300,
    secrets=secrets,
    retries=2,
)
def process_image(
    image_bytes: bytes,
    celebrant: str,
    photographer: str,
    filename: str,
    celebration_id: str,
) -> dict:
    """
    Process a single image: upload to S3, detect faces, store in DB.
    This is the Modal equivalent of _handle_single_upload.
    """
    import os
    import uuid
    import json
    import hashlib
    import logging
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db = get_db_session()
    s3 = get_s3_client()
    redis_client = get_redis_client()
    bucket = os.environ.get("AWS_S3_BUCKET")

    # Import models (need to define inline for Modal)
    from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ForeignKey, ARRAY
    from sqlalchemy.dialects.postgresql import UUID as PGUUID
    from sqlalchemy.orm import declarative_base, relationship
    import uuid as uuid_lib
    from datetime import datetime

    Base = declarative_base()

    class WeddingImage(Base):
        __tablename__ = "wedding_images"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        filename = Column(String, nullable=False)
        file_path = Column(String, nullable=False)
        compressed_file_path = Column(String)
        file_hash = Column(String, unique=True)
        upload_date = Column(DateTime, default=datetime.utcnow)
        faces_count = Column(Integer, default=0)
        processed = Column(String, default="pending")
        quality_analyzed = Column(Boolean, default=False)
        order_number = Column(Integer)

    from pgvector.sqlalchemy import Vector

    class FaceVector(Base):
        __tablename__ = "face_vectors"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        image_id = Column(PGUUID(as_uuid=True), nullable=False)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        face_index = Column(Integer, nullable=False)
        vector = Column(ARRAY(Float), nullable=False)
        vector_pg = Column(Vector(512))
        bbox = Column(ARRAY(Float))
        landmarks = Column(ARRAY(Float))
        confidence = Column(Float)
        quality_score = Column(Float)
        embedding_model = Column(String(40))
        created_date = Column(DateTime, default=datetime.utcnow)

    try:
        if not image_bytes:
            logger.warning(f"Empty content for {filename}")
            return {"status": "skipped", "reason": "empty_content"}

        # Calculate hash
        file_hash = hashlib.sha256(image_bytes).hexdigest()

        # Check for duplicate
        existing = db.query(WeddingImage).filter(WeddingImage.file_hash == file_hash).first()
        if existing:
            logger.info(f"Skipped duplicate file {filename}")
            return {"status": "skipped", "reason": "duplicate", "file_hash": file_hash}

        # Upload to S3 (single copy — files are already optimized at upload time)
        orig_key = f"{photographer}/{celebrant}/{uuid.uuid4()}_{filename}"
        s3.put_object(
            Bucket=bucket,
            Key=orig_key,
            Body=image_bytes,
            ContentType="image/jpeg",
            ACL="public-read",
        )
        s3_endpoint = os.environ.get("S3_ENDPOINT", "")
        if s3_endpoint:
            from urllib.parse import urlparse
            host = urlparse(s3_endpoint).netloc
            url = f"https://{bucket}.{host}/{orig_key}"
        else:
            url = f"https://{bucket}.s3.amazonaws.com/{orig_key}"

        # Create DB record — both paths point to the same file
        img = WeddingImage(
            filename=filename,
            file_path=url,
            compressed_file_path=url,
            file_hash=file_hash,
            processed="processing",
            celebration_id=uuid.UUID(celebration_id),
        )
        db.add(img)
        db.commit()
        db.refresh(img)

        logger.info(f"Added {filename}, starting face detection...")

        image_bgr = _decode_image_with_exif(image_bytes)

        if image_bgr is None:
            img.processed = "failed"
            db.commit()
            return {"status": "failed", "reason": "decode_error"}

        # Initialize face model
        face_app = FaceAnalysis(name=INSIGHTFACE_MODEL)
        face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))

        # Detect faces
        faces = face_app.get(image_bgr)
        face_data = []
        out_index = 0

        for f in faces:
            if f.embedding is None or len(f.embedding) != 512:
                continue

            x1, y1, x2, y2 = f.bbox.astype(int)
            face_w = max(x2 - x1, 0)
            face_h = max(y2 - y1, 0)
            if min(face_w, face_h) < MIN_FACE_PIXELS:
                continue

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

            embedding = f.embedding.tolist()
            face_record = FaceVector(
                image_id=img.id,
                celebration_id=img.celebration_id,
                face_index=out_index,
                vector=embedding,
                vector_pg=embedding,
                bbox=f.bbox.tolist(),
                landmarks=f.kps.flatten().tolist(),
                confidence=float(f.det_score),
                quality_score=quality,
                embedding_model=EMBEDDING_MODEL_VERSION,
            )
            db.add(face_record)

            face_data.append({
                "face_index": out_index,
                "vector": f.embedding.tolist(),
                "bbox": f.bbox.tolist(),
                "landmarks": f.kps.flatten().tolist(),
                "confidence": float(f.det_score),
                "quality_score": quality,
            })
            out_index += 1

        img.faces_count = len(face_data)
        img.processed = "completed"
        db.commit()

        # Cache in Redis
        redis_client.setex(f"image_faces:{img.id}", 3600, json.dumps(face_data, default=str))

        logger.info(f"Processed {len(face_data)} faces for {filename}")

        return {
            "status": "completed",
            "image_id": str(img.id),
            "faces_count": len(face_data),
            "file_hash": file_hash,
        }

    except Exception as e:
        logger.exception(f"Failed to handle {filename}: {e}")
        db.rollback()
        return {"status": "failed", "reason": str(e)}
    finally:
        db.close()


@app.function(
    memory=1024,
    cpu=1.0,
    timeout=120,
    secrets=secrets,
    retries=1,
)
def analyze_single_image(
    image_id: str,
    file_path: str,
    threshold: float = 0.70,
) -> dict:
    """Analyze a single image for quality issues. Called in parallel."""
    import os
    import logging
    import cv2
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    s3 = get_s3_client()
    bucket = os.environ.get("AWS_S3_BUCKET")

    # Quality detection functions
    def detect_blur(image, thresh=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = var < thresh
        conf = 1.0 - min(var / thresh, 1.0) if is_blurry else 0.0
        return is_blurry, conf

    def detect_motion_blur(image, thresh=0.7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = mag.shape
        ch, cw = h // 2, w // 2
        hband = mag[ch-5:ch+5, :].sum()
        vband = mag[:, cw-5:cw+5].sum()
        ratio = max(hband, vband) / (min(hband, vband) + 1e-10)
        has_blur = ratio > (1 / thresh) if thresh > 0 else False
        conf = min(ratio / 3.0, 1.0) if has_blur else 0.0
        return has_blur, conf

    def detect_underexposed(image, bright_thresh=50.0, dark_ratio_thresh=0.5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_bright = gray.mean()
        dark_ratio = np.sum(gray < 30) / gray.size
        is_under = mean_bright < bright_thresh or dark_ratio > dark_ratio_thresh
        if is_under:
            conf = max(1.0 - min(mean_bright / bright_thresh, 1.0),
                      min(dark_ratio / dark_ratio_thresh, 1.0) if dark_ratio_thresh > 0 else 0)
        else:
            conf = 0.0
        return is_under, conf

    def detect_overexposed(image, bright_thresh=205.0, clip_thresh=0.1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_bright = gray.mean()
        clip_ratio = np.sum(gray > 250) / gray.size
        is_over = mean_bright > bright_thresh or clip_ratio > clip_thresh
        if is_over:
            bf = min((mean_bright - bright_thresh) / (255 - bright_thresh), 1.0) if mean_bright > bright_thresh else 0
            rf = min(clip_ratio / clip_thresh, 1.0) if clip_thresh > 0 else 0
            conf = max(bf, rf)
        else:
            conf = 0.0
        return is_over, conf

    try:
        key = extract_s3_key(file_path)
        resp = s3.get_object(Bucket=bucket, Key=key)
        img_bytes = resp["Body"].read()

        nparr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return {"image_id": image_id, "issues": [], "error": "decode_failed"}

        issues = []

        is_blurry, blur_conf = detect_blur(image_bgr)
        if is_blurry and blur_conf >= threshold:
            issues.append(("blur", blur_conf))

        has_motion, motion_conf = detect_motion_blur(image_bgr)
        if has_motion and motion_conf >= threshold:
            issues.append(("motion_blur", motion_conf))

        is_under, under_conf = detect_underexposed(image_bgr)
        if is_under and under_conf >= threshold:
            issues.append(("underexposed", under_conf))

        is_over, over_conf = detect_overexposed(image_bgr)
        if is_over and over_conf >= threshold:
            issues.append(("overexposed", over_conf))

        return {"image_id": image_id, "issues": issues, "error": None}

    except Exception as e:
        logger.error(f"Error analyzing {image_id}: {e}")
        return {"image_id": image_id, "issues": [], "error": str(e)}


@app.function(
    memory=512,
    cpu=0.5,
    timeout=900,
    secrets=secrets,
    retries=1,
)
def analyze_quality(
    celebration_id: str,
    threshold: float = 0.70,
    reanalyze: bool = False,
) -> dict:
    """
    Analyze all images in a celebration for quality issues.
    Uses parallel processing - each image analyzed in separate container.
    """
    import os
    import uuid
    import logging
    from datetime import datetime

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db = get_db_session()

    # Import models inline
    from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ForeignKey, ARRAY
    from sqlalchemy.dialects.postgresql import UUID as PGUUID
    from sqlalchemy.orm import declarative_base
    import uuid as uuid_lib

    Base = declarative_base()

    class Celebration(Base):
        __tablename__ = "celebrations"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        celebrant = Column(String, nullable=False)
        photographer = Column(String, nullable=False)

    class WeddingImage(Base):
        __tablename__ = "wedding_images"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        filename = Column(String, nullable=False)
        file_path = Column(String, nullable=False)
        compressed_file_path = Column(String)
        quality_analyzed = Column(Boolean, default=False)

    class FaceVector(Base):
        __tablename__ = "face_vectors"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        image_id = Column(PGUUID(as_uuid=True), nullable=False)
        landmarks = Column(ARRAY(Float))
        bbox = Column(ARRAY(Float))

    class QualityAnalysisJob(Base):
        __tablename__ = "quality_analysis_jobs"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        total_images = Column(Integer, default=0)
        processed_count = Column(Integer, default=0)
        flagged_count = Column(Integer, default=0)
        status = Column(String, default="pending")
        threshold = Column(Float, default=0.70)
        started_at = Column(DateTime, default=datetime.utcnow)
        completed_at = Column(DateTime)
        error_message = Column(String)

    class ImageQualityFlag(Base):
        __tablename__ = "image_quality_flags"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        image_id = Column(PGUUID(as_uuid=True), nullable=False)
        issue_type = Column(String, nullable=False)
        confidence = Column(Float, nullable=False)
        reviewed = Column(Boolean, default=False)
        dismissed = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.utcnow)

    # Quality detection functions (inline)
    def detect_blur(image, thresh=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = var < thresh
        conf = 1.0 - min(var / thresh, 1.0) if is_blurry else 0.0
        return is_blurry, conf

    def detect_motion_blur(image, thresh=0.7):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = mag.shape
        ch, cw = h // 2, w // 2
        hband = mag[ch-5:ch+5, :].sum()
        vband = mag[:, cw-5:cw+5].sum()
        ratio = max(hband, vband) / (min(hband, vband) + 1e-10)
        has_blur = ratio > (1 / thresh) if thresh > 0 else False
        conf = min(ratio / 3.0, 1.0) if has_blur else 0.0
        return has_blur, conf

    def detect_underexposed(image, bright_thresh=50.0, dark_ratio_thresh=0.5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_bright = gray.mean()
        dark_ratio = np.sum(gray < 30) / gray.size
        is_under = mean_bright < bright_thresh or dark_ratio > dark_ratio_thresh
        if is_under:
            conf = max(1.0 - min(mean_bright / bright_thresh, 1.0),
                      min(dark_ratio / dark_ratio_thresh, 1.0) if dark_ratio_thresh > 0 else 0)
        else:
            conf = 0.0
        return is_under, conf

    def detect_overexposed(image, bright_thresh=205.0, clip_thresh=0.1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_bright = gray.mean()
        clip_ratio = np.sum(gray > 250) / gray.size
        is_over = mean_bright > bright_thresh or clip_ratio > clip_thresh
        if is_over:
            bf = min((mean_bright - bright_thresh) / (255 - bright_thresh), 1.0) if mean_bright > bright_thresh else 0
            rf = min(clip_ratio / clip_thresh, 1.0) if clip_thresh > 0 else 0
            conf = max(bf, rf)
        else:
            conf = 0.0
        return is_over, conf

    # Initialize S3 client and imports
    import cv2
    import numpy as np
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    s3 = get_s3_client()
    bucket = os.environ.get("AWS_S3_BUCKET")
    S3_WORKERS = 8

    def _decode(image_bytes: bytes):
        try:
            import io
            from PIL import Image, ImageOps
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except Exception:
                pass
            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _calibrate(sample_paths):
        """Sample-driven percentile thresholds; falls back to defaults on tiny events."""
        if len(sample_paths) < 5:
            return None
        laps, brights = [], []

        def _stat(fp):
            try:
                k = extract_s3_key(fp)
                data = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
                im = _decode(data)
                if im is None:
                    return None
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=S3_WORKERS) as pool:
            for res in pool.map(_stat, sample_paths):
                if res:
                    laps.append(res[0])
                    brights.append(res[1])
        if len(laps) < 5:
            return None
        laps.sort()
        brights.sort()
        n = len(laps)
        def pct(arr, q):
            return arr[max(0, min(n - 1, int(q * (n - 1))))]
        return {
            "blur": max(50.0, min(180.0, pct(laps, 0.25))),
            "brightness_low": max(15.0, min(60.0, pct(brights, 0.10))),
            "brightness_high": max(180.0, min(240.0, pct(brights, 0.90))),
            "n": n,
        }

    try:
        celeb_uuid = uuid.UUID(celebration_id)

        # Get images to analyze
        query = db.query(WeddingImage).filter(WeddingImage.celebration_id == celeb_uuid)
        if not reanalyze:
            query = query.filter(WeddingImage.quality_analyzed == False)
        images = query.all()

        # Reuse the pending job the API created on trigger instead of inserting
        # a new row. Otherwise we end up with stale orphan jobs that block
        # future triggers and confuse the status endpoint.
        job = (
            db.query(QualityAnalysisJob)
            .filter(
                QualityAnalysisJob.celebration_id == celeb_uuid,
                QualityAnalysisJob.status.in_(["pending", "processing"]),
            )
            .order_by(QualityAnalysisJob.started_at.desc())
            .first()
        )
        if job is None:
            job = QualityAnalysisJob(
                celebration_id=celeb_uuid,
                total_images=len(images),
                status="processing",
                threshold=threshold,
            )
            db.add(job)
        else:
            job.total_images = len(images)
            job.status = "processing"
            job.threshold = threshold
            job.error_message = None
            job.completed_at = None
        db.commit()
        db.refresh(job)

        if not images:
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            db.commit()
            return {"status": "completed", "processed": 0, "flagged": 0}

        # Snapshot ORM fields into plain tuples so worker threads never touch
        # the SQLAlchemy session (sessions are not thread-safe; lazy loads from
        # background threads were causing inner-flush warnings and duplicate
        # inserts).
        image_records = [
            (img.id, img.compressed_file_path or img.file_path) for img in images
        ]
        images_by_id = {img.id: img for img in images}

        # Re-runs must be idempotent: clear any existing flags for the images
        # we're about to re-analyze, otherwise reanalyze=True would race against
        # itself if a previous run partially wrote flags.
        if reanalyze:
            image_ids = [r[0] for r in image_records]
            db.query(ImageQualityFlag).filter(
                ImageQualityFlag.image_id.in_(image_ids)
            ).delete(synchronize_session=False)
            db.commit()

        # Calibrate per-celebration thresholds
        sample_n = min(25, len(image_records))
        sample_paths = (
            [r[1] for r in random.sample(image_records, sample_n)]
            if len(image_records) > sample_n
            else [r[1] for r in image_records]
        )
        cal = _calibrate(sample_paths)
        if cal:
            logger.info(f"Calibrated (n={cal['n']}): blur={cal['blur']:.1f} under<{cal['brightness_low']:.1f} over>{cal['brightness_high']:.1f}")
        blur_t = (cal or {}).get("blur", 100.0)
        under_t = (cal or {}).get("brightness_low", 30.0)
        over_t = (cal or {}).get("brightness_high", 220.0)

        processed = 0
        flagged = 0

        def _fetch(record):
            img_id, fp = record
            try:
                k = extract_s3_key(fp)
                return img_id, s3.get_object(Bucket=bucket, Key=k)["Body"].read(), None
            except Exception as e:
                return img_id, None, str(e)

        with ThreadPoolExecutor(max_workers=S3_WORKERS) as pool:
            futures = [pool.submit(_fetch, rec) for rec in image_records]

            for future in as_completed(futures):
                img_id, img_bytes, err = future.result()
                try:
                    if err:
                        logger.warning(f"Fetch failed for {img_id}: {err}")
                        processed += 1
                        continue

                    image_bgr = _decode(img_bytes) if img_bytes else None
                    if image_bgr is None:
                        processed += 1
                        continue

                    issues = []

                    is_blurry, blur_conf = detect_blur(image_bgr, thresh=blur_t)
                    if is_blurry and blur_conf >= threshold:
                        issues.append(("blur", blur_conf))

                    has_motion, motion_conf = detect_motion_blur(image_bgr)
                    if has_motion and motion_conf >= threshold:
                        issues.append(("motion_blur", motion_conf))

                    is_under, under_conf = detect_underexposed(image_bgr, bright_thresh=under_t)
                    if is_under and under_conf >= threshold:
                        issues.append(("underexposed", under_conf))

                    is_over, over_conf = detect_overexposed(image_bgr, bright_thresh=over_t)
                    if is_over and over_conf >= threshold:
                        issues.append(("overexposed", over_conf))

                    if issues:
                        flagged += 1
                        for issue_type, conf in issues:
                            db.add(ImageQualityFlag(
                                image_id=img_id,
                                issue_type=issue_type,
                                confidence=conf,
                            ))

                    img_row = images_by_id.get(img_id)
                    if img_row is not None:
                        img_row.quality_analyzed = True
                    processed += 1

                    if processed % 10 == 0:
                        job.processed_count = processed
                        job.flagged_count = flagged
                        db.commit()
                except Exception as e:
                    # Roll back so the session can keep being used. Without this,
                    # the next attribute access triggers PendingRollbackError.
                    db.rollback()
                    logger.error(f"Error analyzing {img_id}: {e}")
                    continue

        job.processed_count = processed
        job.flagged_count = flagged
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Quality analysis complete: {processed} images, {flagged} flagged")
        return {"status": "completed", "processed": processed, "flagged": flagged, "job_id": str(job.id)}

    except Exception as e:
        logger.exception(f"Quality analysis failed: {e}")
        # Mark the job as failed so the frontend stops spinning on
        # "جاري التحليل" forever. The session may be in a bad state, so roll
        # back first and re-query.
        try:
            db.rollback()
            stuck_job = (
                db.query(QualityAnalysisJob)
                .filter(
                    QualityAnalysisJob.celebration_id == uuid.UUID(celebration_id),
                    QualityAnalysisJob.status.in_(["pending", "processing"]),
                )
                .order_by(QualityAnalysisJob.started_at.desc())
                .first()
            )
            if stuck_job is not None:
                stuck_job.status = "failed"
                stuck_job.error_message = str(e)[:500]
                stuck_job.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            logger.exception("Failed to record analysis failure on job row")
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()


@app.function(
    memory=2048,
    cpu=2.0,
    timeout=300,
    secrets=secrets,
)
def reprocess_image(image_id: str) -> dict:
    """Reprocess a single image for face detection."""
    import os
    import uuid
    import json
    import logging
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    db = get_db_session()
    s3 = get_s3_client()
    redis_client = get_redis_client()
    bucket = os.environ.get("AWS_S3_BUCKET")

    from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ARRAY
    from sqlalchemy.dialects.postgresql import UUID as PGUUID
    from sqlalchemy.orm import declarative_base
    import uuid as uuid_lib
    from datetime import datetime

    Base = declarative_base()

    class WeddingImage(Base):
        __tablename__ = "wedding_images"
        id = Column(PGUUID(as_uuid=True), primary_key=True)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        filename = Column(String, nullable=False)
        file_path = Column(String, nullable=False)
        compressed_file_path = Column(String)
        faces_count = Column(Integer, default=0)
        processed = Column(String, default="pending")

    from pgvector.sqlalchemy import Vector

    class FaceVector(Base):
        __tablename__ = "face_vectors"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        image_id = Column(PGUUID(as_uuid=True), nullable=False)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        face_index = Column(Integer, nullable=False)
        vector = Column(ARRAY(Float), nullable=False)
        vector_pg = Column(Vector(512))
        bbox = Column(ARRAY(Float))
        landmarks = Column(ARRAY(Float))
        confidence = Column(Float)
        quality_score = Column(Float)
        embedding_model = Column(String(40))
        created_date = Column(DateTime, default=datetime.utcnow)

    try:
        img_uuid = uuid.UUID(image_id)
        img = db.query(WeddingImage).filter(WeddingImage.id == img_uuid).first()

        if not img:
            return {"status": "failed", "reason": "image_not_found"}

        # Always use the original image so bbox coordinates match what the frontend displays
        file_path = img.file_path
        key = extract_s3_key(file_path)

        resp = s3.get_object(Bucket=bucket, Key=key)
        img_bytes = resp["Body"].read()

        image_bgr = _decode_image_with_exif(img_bytes)

        if image_bgr is None:
            img.processed = "failed"
            db.commit()
            return {"status": "failed", "reason": "decode_error"}

        # Delete old face vectors
        db.query(FaceVector).filter(FaceVector.image_id == img.id).delete()

        # Detect faces
        face_app = FaceAnalysis(name=INSIGHTFACE_MODEL)
        face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
        faces = face_app.get(image_bgr)

        face_data = []
        out_index = 0
        for f in faces:
            if f.embedding is None or len(f.embedding) != 512:
                continue

            x1, y1, x2, y2 = f.bbox.astype(int)
            face_w = max(x2 - x1, 0)
            face_h = max(y2 - y1, 0)
            if min(face_w, face_h) < MIN_FACE_PIXELS:
                continue

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

            embedding = f.embedding.tolist()
            face_record = FaceVector(
                image_id=img.id,
                celebration_id=img.celebration_id,
                face_index=out_index,
                vector=embedding,
                vector_pg=embedding,
                bbox=f.bbox.tolist(),
                landmarks=f.kps.flatten().tolist(),
                confidence=float(f.det_score),
                quality_score=quality,
                embedding_model=EMBEDDING_MODEL_VERSION,
            )
            db.add(face_record)
            face_data.append({
                "face_index": out_index,
                "bbox": f.bbox.tolist(),
                "confidence": float(f.det_score),
                "quality_score": quality,
            })
            out_index += 1

        img.faces_count = len(face_data)
        img.processed = "completed"
        db.commit()

        redis_client.setex(f"image_faces:{img.id}", 3600, json.dumps(face_data, default=str))

        logger.info(f"Reprocessed {img.filename}: {len(face_data)} faces")
        return {"status": "completed", "image_id": image_id, "faces_count": len(face_data)}

    except Exception as e:
        logger.exception(f"Reprocess failed: {e}")
        return {"status": "failed", "reason": str(e)}
    finally:
        db.close()


@app.local_entrypoint()
def main():
    """Test the worker locally."""
    print("Modal worker ready. Deploy with: modal deploy modal_worker.py")
    print("\nAvailable functions:")
    print("  - process_image: Process uploaded images")
    print("  - analyze_quality: Analyze celebration for quality issues")
    print("  - reprocess_image: Reprocess a single image")
