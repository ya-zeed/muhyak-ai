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
        "boto3>=1.34,<2.0",
        "redis>=5.0,<6.0",
        "psycopg[binary]>=3.1,<3.2",
        "sqlalchemy>=2.0,<2.1",
    )
    .run_commands(
        # Pre-download the fast model
        "python -c \""
        "from insightface.app import FaceAnalysis; "
        "a = FaceAnalysis(name='buffalo_s'); "
        "a.prepare(ctx_id=-1, det_size=(320,320)); "
        "print('Model cached')\""
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
    from PIL import Image
    from io import BytesIO
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

    class FaceVector(Base):
        __tablename__ = "face_vectors"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        image_id = Column(PGUUID(as_uuid=True), nullable=False)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        face_index = Column(Integer, nullable=False)
        vector = Column(ARRAY(Float), nullable=False)
        bbox = Column(ARRAY(Float))
        landmarks = Column(ARRAY(Float))
        confidence = Column(Float)
        quality_score = Column(Float)
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

        # Upload original to S3
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
            orig_url = f"https://{bucket}.{host}/{orig_key}"
        else:
            orig_url = f"https://{bucket}.s3.amazonaws.com/{orig_key}"

        # Compress image
        pil_img = Image.open(BytesIO(image_bytes))
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        comp_buffer = BytesIO()
        pil_img.save(comp_buffer, format="JPEG", quality=75)
        comp_bytes = comp_buffer.getvalue()

        # Upload compressed
        comp_key = f"{photographer}/{celebrant}/compressed_{uuid.uuid4()}.jpg"
        s3.put_object(
            Bucket=bucket,
            Key=comp_key,
            Body=comp_bytes,
            ContentType="image/jpeg",
            ACL="public-read",
        )
        if s3_endpoint:
            comp_url = f"https://{bucket}.{host}/{comp_key}"
        else:
            comp_url = f"https://{bucket}.s3.amazonaws.com/{comp_key}"

        # Create DB record
        img = WeddingImage(
            filename=filename,
            file_path=orig_url,
            compressed_file_path=comp_url,
            file_hash=file_hash,
            processed="processing",
            celebration_id=uuid.UUID(celebration_id),
        )
        db.add(img)
        db.commit()
        db.refresh(img)

        logger.info(f"Added {filename}, starting face detection...")

        # Initialize face model
        face_app = FaceAnalysis(name="buffalo_s")
        face_app.prepare(ctx_id=0, det_size=(320, 320))

        # Decode image for face detection
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            img.processed = "failed"
            db.commit()
            return {"status": "failed", "reason": "decode_error"}

        # Detect faces
        faces = face_app.get(image_bgr)
        face_data = []

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

            face_record = FaceVector(
                image_id=img.id,
                celebration_id=img.celebration_id,
                face_index=i,
                vector=f.embedding.tolist(),
                bbox=f.bbox.tolist(),
                landmarks=f.kps.flatten().tolist(),
                confidence=float(f.det_score),
                quality_score=quality,
            )
            db.add(face_record)

            face_data.append({
                "face_index": i,
                "vector": f.embedding.tolist(),
                "bbox": f.bbox.tolist(),
                "landmarks": f.kps.flatten().tolist(),
                "confidence": float(f.det_score),
                "quality_score": quality,
            })

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

    try:
        celeb_uuid = uuid.UUID(celebration_id)

        # Get images to analyze
        query = db.query(WeddingImage).filter(WeddingImage.celebration_id == celeb_uuid)
        if not reanalyze:
            query = query.filter(WeddingImage.quality_analyzed == False)
        images = query.all()

        # Create/update job
        job = QualityAnalysisJob(
            celebration_id=celeb_uuid,
            total_images=len(images),
            status="processing",
            threshold=threshold,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        if not images:
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            db.commit()
            return {"status": "completed", "processed": 0, "flagged": 0}

        processed = 0
        flagged = 0

        for img in images:
            try:
                file_path = img.compressed_file_path or img.file_path
                key = extract_s3_key(file_path)

                resp = s3.get_object(Bucket=bucket, Key=key)
                img_bytes = resp["Body"].read()

                nparr = np.frombuffer(img_bytes, np.uint8)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image_bgr is None:
                    processed += 1
                    continue

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

                if issues:
                    flagged += 1
                    for issue_type, conf in issues:
                        flag = ImageQualityFlag(
                            image_id=img.id,
                            issue_type=issue_type,
                            confidence=conf,
                        )
                        db.add(flag)

                img.quality_analyzed = True
                processed += 1

                if processed % 10 == 0:
                    job.processed_count = processed
                    job.flagged_count = flagged
                    db.commit()

            except Exception as e:
                logger.error(f"Error analyzing {img.id}: {e}")
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

    class FaceVector(Base):
        __tablename__ = "face_vectors"
        id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
        image_id = Column(PGUUID(as_uuid=True), nullable=False)
        celebration_id = Column(PGUUID(as_uuid=True), nullable=False)
        face_index = Column(Integer, nullable=False)
        vector = Column(ARRAY(Float), nullable=False)
        bbox = Column(ARRAY(Float))
        landmarks = Column(ARRAY(Float))
        confidence = Column(Float)
        quality_score = Column(Float)
        created_date = Column(DateTime, default=datetime.utcnow)

    try:
        img_uuid = uuid.UUID(image_id)
        img = db.query(WeddingImage).filter(WeddingImage.id == img_uuid).first()

        if not img:
            return {"status": "failed", "reason": "image_not_found"}

        # Download from S3
        file_path = img.compressed_file_path or img.file_path
        key = extract_s3_key(file_path)

        resp = s3.get_object(Bucket=bucket, Key=key)
        img_bytes = resp["Body"].read()

        nparr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            img.processed = "failed"
            db.commit()
            return {"status": "failed", "reason": "decode_error"}

        # Delete old face vectors
        db.query(FaceVector).filter(FaceVector.image_id == img.id).delete()

        # Detect faces
        face_app = FaceAnalysis(name="buffalo_s")
        face_app.prepare(ctx_id=0, det_size=(320, 320))
        faces = face_app.get(image_bgr)

        face_data = []
        for i, f in enumerate(faces):
            if f.embedding is None or len(f.embedding) != 512:
                continue

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

            face_record = FaceVector(
                image_id=img.id,
                celebration_id=img.celebration_id,
                face_index=i,
                vector=f.embedding.tolist(),
                bbox=f.bbox.tolist(),
                landmarks=f.kps.flatten().tolist(),
                confidence=float(f.det_score),
                quality_score=quality,
            )
            db.add(face_record)
            face_data.append({
                "face_index": i,
                "bbox": f.bbox.tolist(),
                "confidence": float(f.det_score),
                "quality_score": quality,
            })

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
