"""
Quality Analyzer Service - AI Image Quality Detection

T007-T013: Detection functions for blur, motion blur, closed eyes, exposure issues
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from sqlalchemy.orm import Session

from models import WeddingImage, ImageQualityFlag, QualityAnalysisJob, Celebration

logger = logging.getLogger(__name__)


# T007: Blur detection using Laplacian variance
def detect_blur(image: np.ndarray, threshold: float = 100.0) -> tuple[bool, float]:
    """
    Detect blur/out-of-focus images using Laplacian variance.

    Args:
        image: BGR image array from OpenCV
        threshold: Variance threshold below which image is considered blurry

    Returns:
        tuple of (is_blurry, confidence_score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    is_blurry = laplacian_var < threshold
    # Confidence: higher when more certain about blur
    if is_blurry:
        # Low variance = high confidence of blur
        confidence = 1.0 - min(laplacian_var / threshold, 1.0)
    else:
        confidence = 0.0

    return is_blurry, confidence


# T008: Motion blur detection using FFT analysis
def detect_motion_blur(image: np.ndarray, threshold: float = 0.7) -> tuple[bool, float]:
    """
    Detect motion blur using FFT (Fourier Transform) analysis.
    Motion blur creates directional patterns in frequency domain.

    Args:
        image: BGR image array from OpenCV
        threshold: Ratio threshold for directional blur detection

    Returns:
        tuple of (has_motion_blur, confidence_score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Analyze directional energy distribution
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # Horizontal vs vertical energy bands
    horizontal_band = magnitude[center_h-5:center_h+5, :].sum()
    vertical_band = magnitude[:, center_w-5:center_w+5].sum()

    # Motion blur creates elongated patterns
    ratio = max(horizontal_band, vertical_band) / (min(horizontal_band, vertical_band) + 1e-10)

    has_motion_blur = ratio > (1 / threshold) if threshold > 0 else False
    confidence = min(ratio / 3.0, 1.0) if has_motion_blur else 0.0

    return has_motion_blur, confidence


# T009: Closed eyes detection
#
# InsightFace's `buffalo_l` returns only 5 landmark points (left_eye, right_eye,
# nose, left_mouth, right_mouth) — single centers, not an eye contour. So a
# classic Eye Aspect Ratio formula isn't computable directly.
#
# Instead, we crop an eye-sized patch around each eye center (scaled by the
# inter-ocular distance) and score it on two skin-tone-robust signals:
#
#   1. Laplacian variance — texture/edge content of the patch. Open eyes have
#      sharp iris/pupil/eyelash edges; closed eyes show only smooth eyelid skin.
#   2. Vertical intensity range — open eyes show a wide light/dark range
#      (sclera + pupil); closed eyes are uniformly toned.
#
# These beat the previous "mean intensity < 80" heuristic which silently
# false-positives on darker skin tones and false-negatives on bright backlight.

# Patch size as a fraction of inter-ocular distance. Calibrated against typical
# face geometry (eye width ≈ 0.45 IOD, eye height ≈ 0.30 IOD).
_EYE_PATCH_HALF_W = 0.225
_EYE_PATCH_HALF_H = 0.15

# Score thresholds. Both edge-density and intensity-range have to look "flat"
# for an eye to be called closed. Conservative on purpose — flagging an awake
# guest as "closed eyes" is much worse than missing a blink in one frame.
_EDGE_BASELINE = 200.0  # Laplacian variance considered "definitely open"
_RANGE_BASELINE = 80.0  # intensity range considered "definitely open"
_CLOSED_SCORE_THRESHOLD = 0.55


def detect_closed_eyes(image: np.ndarray, faces: list, ear_threshold: float = 0.2) -> tuple[bool, float]:
    """Detect closed eyes from per-face landmarks.

    ``ear_threshold`` is retained for API compatibility but unused; the
    classifier now scores patch texture + intensity range instead of EAR.
    Returns the maximum closed-score across faces in the image.
    """
    if not faces:
        return False, 0.0

    h, w = image.shape[:2]
    max_closed_score = 0.0

    def _patch_closed_score(crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        edge_score = 1.0 - min(cv2.Laplacian(gray, cv2.CV_64F).var() / _EDGE_BASELINE, 1.0)
        intensity_range = float(gray.max()) - float(gray.min())
        range_score = 1.0 - min(intensity_range / _RANGE_BASELINE, 1.0)
        # Both signals have to agree — geometric mean punishes
        # cases where only one fires.
        return float((edge_score * range_score) ** 0.5)

    for face in faces:
        landmarks = face.get('landmarks') or face.get('kps')
        if landmarks is None:
            continue

        landmarks = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if len(landmarks) < 2:
            continue

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        inter_ocular = float(np.linalg.norm(left_eye - right_eye))
        if inter_ocular < 8:
            # Face too small to read eye state reliably; skip rather than guess.
            continue

        half_w = max(int(inter_ocular * _EYE_PATCH_HALF_W), 4)
        half_h = max(int(inter_ocular * _EYE_PATCH_HALF_H), 3)

        eye_scores: list[float] = []
        for eye_xy in (left_eye, right_eye):
            x, y = int(eye_xy[0]), int(eye_xy[1])
            x0 = max(x - half_w, 0)
            x1 = min(x + half_w, w)
            y0 = max(y - half_h, 0)
            y1 = min(y + half_h, h)
            crop = image[y0:y1, x0:x1]
            eye_scores.append(_patch_closed_score(crop))

        if not eye_scores:
            continue

        # Average of both eyes — winks aren't "closed eyes" for photo-cull purposes.
        face_score = sum(eye_scores) / len(eye_scores)
        if face_score > _CLOSED_SCORE_THRESHOLD:
            max_closed_score = max(max_closed_score, face_score)

    return (max_closed_score > _CLOSED_SCORE_THRESHOLD, max_closed_score)


# T010: Underexposure detection using histogram analysis
def detect_underexposed(image: np.ndarray, brightness_threshold: float = 30.0, dark_ratio_threshold: float = 0.7) -> tuple[bool, float]:
    """
    Detect underexposed (too dark) images using histogram analysis.

    Args:
        image: BGR image array from OpenCV
        brightness_threshold: Mean brightness below which image is underexposed
        dark_ratio_threshold: Ratio of dark pixels that indicates underexposure

    Returns:
        tuple of (is_underexposed, confidence_score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    # Also check ratio of very dark pixels
    dark_pixels = np.sum(gray < 30)
    total_pixels = gray.size
    dark_ratio = dark_pixels / total_pixels

    is_underexposed = mean_brightness < brightness_threshold or dark_ratio > dark_ratio_threshold

    if is_underexposed:
        # Higher confidence when darker or more dark pixels
        brightness_factor = 1.0 - min(mean_brightness / brightness_threshold, 1.0)
        ratio_factor = min(dark_ratio / dark_ratio_threshold, 1.0) if dark_ratio_threshold > 0 else 0
        confidence = max(brightness_factor, ratio_factor)
    else:
        confidence = 0.0

    return is_underexposed, confidence


# T011: Overexposure detection using histogram analysis
def detect_overexposed(image: np.ndarray, brightness_threshold: float = 220.0, clipped_threshold: float = 0.25) -> tuple[bool, float]:
    """
    Detect overexposed (too bright/washed out) images.

    Args:
        image: BGR image array from OpenCV
        brightness_threshold: Mean brightness above which image is overexposed
        clipped_threshold: Ratio of clipped (max brightness) pixels indicating overexposure

    Returns:
        tuple of (is_overexposed, confidence_score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    # Check for clipped highlights
    clipped_pixels = np.sum(gray > 250)
    total_pixels = gray.size
    clipped_ratio = clipped_pixels / total_pixels

    is_overexposed = mean_brightness > brightness_threshold or clipped_ratio > clipped_threshold

    if is_overexposed:
        brightness_factor = min((mean_brightness - brightness_threshold) / (255 - brightness_threshold), 1.0) if mean_brightness > brightness_threshold else 0
        ratio_factor = min(clipped_ratio / clipped_threshold, 1.0) if clipped_threshold > 0 else 0
        confidence = max(brightness_factor, ratio_factor)
    else:
        confidence = 0.0

    return is_overexposed, confidence


# T012: Combine all detectors for single image analysis
def analyze_single_image(
    image: np.ndarray,
    faces: list | None = None,
    threshold: float = 0.70,
    calibrated: dict | None = None,
) -> list[dict]:
    """
    Analyze a single image for all quality issues.

    Args:
        image: BGR image array from OpenCV
        faces: Optional list of detected faces with landmarks
        threshold: Minimum confidence threshold for flagging issues
        calibrated: Optional per-celebration thresholds from ``_calibrate_thresholds``.
            Falls back to global defaults when missing.

    Returns:
        List of detected issues with type and confidence
    """
    issues = []
    cal = calibrated or {}
    blur_t = cal.get("blur", 100.0)
    under_t = cal.get("brightness_low", 30.0)
    over_t = cal.get("brightness_high", 220.0)

    # Run all detectors
    is_blurry, blur_conf = detect_blur(image, threshold=blur_t)
    if is_blurry and blur_conf >= threshold:
        issues.append({"issue_type": "blur", "confidence": blur_conf})

    has_motion_blur, motion_conf = detect_motion_blur(image)
    if has_motion_blur and motion_conf >= threshold:
        issues.append({"issue_type": "motion_blur", "confidence": motion_conf})

    if faces:
        has_closed_eyes, eyes_conf = detect_closed_eyes(image, faces)
        if has_closed_eyes and eyes_conf >= threshold:
            issues.append({"issue_type": "closed_eyes", "confidence": eyes_conf})

    is_underexposed, under_conf = detect_underexposed(image, brightness_threshold=under_t)
    if is_underexposed and under_conf >= threshold:
        issues.append({"issue_type": "underexposed", "confidence": under_conf})

    is_overexposed, over_conf = detect_overexposed(image, brightness_threshold=over_t)
    if is_overexposed and over_conf >= threshold:
        issues.append({"issue_type": "overexposed", "confidence": over_conf})

    return issues


# Sample size for per-celebration calibration. 25 is enough to estimate
# the 10th/25th/90th percentile of a celebration's brightness/sharpness
# distribution without paying a meaningful extra S3 cost.
_CALIBRATION_SAMPLE = 25
_MIN_CALIBRATION_SAMPLES = 5


def _calibrate_thresholds(images_to_sample, s3_client, bucket: str) -> dict | None:
    """Compute per-celebration percentile-based thresholds.

    Outdoor noon weddings have a totally different luminance profile from
    candle-lit indoor receptions. Flagging issues against fixed
    "underexposed if brightness < 30" thresholds therefore false-positives
    (or false-negatives) all night. We sample a handful of frames, look at
    the distribution, and define "unusually X" relative to *this* event.
    """
    from concurrent.futures import ThreadPoolExecutor
    from utils import load_image_from_bytes

    if not images_to_sample:
        return None

    def _fetch_stats(wedding_image):
        try:
            file_path = wedding_image.compressed_file_path or wedding_image.file_path
            key = _extract_s3_key(file_path, bucket)
            data = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
            try:
                img = load_image_from_bytes(data)
            except Exception:
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return {
                "lap": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "brightness": float(gray.mean()),
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=_S3_FETCH_WORKERS) as pool:
        results = list(pool.map(_fetch_stats, images_to_sample))

    stats = [s for s in results if s is not None]
    if len(stats) < _MIN_CALIBRATION_SAMPLES:
        return None

    laplacians = sorted(s["lap"] for s in stats)
    brightnesses = sorted(s["brightness"] for s in stats)
    n = len(stats)

    def _pct(arr, q: float) -> float:
        idx = max(0, min(n - 1, int(q * (n - 1))))
        return arr[idx]

    # Floors/ceilings keep one extreme image from poisoning the whole event
    # (e.g. a single sunset shot mustn't make the system blind to actual blowouts).
    return {
        "blur": max(50.0, min(180.0, _pct(laplacians, 0.25))),
        "brightness_low": max(15.0, min(60.0, _pct(brightnesses, 0.10))),
        "brightness_high": max(180.0, min(240.0, _pct(brightnesses, 0.90))),
        "sample_size": n,
    }


# How many S3 fetches can be in flight at once.
# 8 saturates a typical Spaces/S3 connection without overwhelming
# the Postgres session or burning Modal memory at ~2MB/image avg.
_S3_FETCH_WORKERS = 8


def _extract_s3_key(file_path: str, bucket: str) -> str:
    """Strip protocol/host/query-string from an S3 URL and return the key."""
    from urllib.parse import urlparse

    parsed = urlparse(file_path)
    key = parsed.path.lstrip("/")
    key = key.split("?")[0]
    if bucket and key.startswith(f"{bucket}/"):
        key = key[len(bucket) + 1:]
    return key


# T013: Batch processing for celebration analysis
def analyze_celebration(
    db: Session,
    celebration_id: uuid.UUID,
    threshold: float = 0.70,
    reanalyze: bool = False
) -> QualityAnalysisJob:
    """
    Analyze all images in a celebration for quality issues.

    Args:
        db: Database session
        celebration_id: UUID of the celebration to analyze
        threshold: Minimum confidence threshold for flagging
        reanalyze: If True, re-analyze previously analyzed images

    Returns:
        QualityAnalysisJob tracking the analysis progress
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import random
    import boto3

    from config import settings
    from utils import load_image_from_bytes

    # Get celebration and images
    celebration = db.query(Celebration).filter(Celebration.id == celebration_id).first()
    if not celebration:
        raise ValueError(f"Celebration {celebration_id} not found")

    # Query images to analyze
    query = db.query(WeddingImage).filter(WeddingImage.celebration_id == celebration_id)
    if not reanalyze:
        query = query.filter(WeddingImage.quality_analyzed == False)

    images = query.all()
    total_images = len(images)

    # Create analysis job
    job = QualityAnalysisJob(
        celebration_id=celebration_id,
        total_images=total_images,
        processed_count=0,
        flagged_count=0,
        status="processing",
        threshold=threshold
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    if total_images == 0:
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()
        return job

    # Setup S3 client for image retrieval
    s3_client = boto3.client(
        's3',
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )
    bucket = settings.AWS_S3_BUCKET

    # Per-celebration calibration. Skip when the event is too small to
    # learn anything statistically meaningful — global defaults still work.
    calibrated = None
    if total_images >= _MIN_CALIBRATION_SAMPLES:
        sample_size = min(_CALIBRATION_SAMPLE, total_images)
        sample = random.sample(images, sample_size) if total_images > sample_size else list(images)
        calibrated = _calibrate_thresholds(sample, s3_client, bucket)
        if calibrated:
            logger.info(
                f"📊 Calibrated thresholds (n={calibrated.get('sample_size')}): "
                f"blur={calibrated['blur']:.1f} "
                f"underexposed<{calibrated['brightness_low']:.1f} "
                f"overexposed>{calibrated['brightness_high']:.1f}"
            )

    def _fetch(wedding_image: WeddingImage) -> tuple[WeddingImage, bytes | None, str | None]:
        try:
            file_path = wedding_image.compressed_file_path or wedding_image.file_path
            key = _extract_s3_key(file_path, bucket)
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            return wedding_image, resp['Body'].read(), None
        except Exception as e:
            return wedding_image, None, str(e)

    flagged_count = 0
    processed_count = 0

    # Fan out S3 fetches; drain as they complete. The decode/analyze/DB-write
    # stays on this thread — SQLAlchemy sessions aren't thread-safe and
    # numpy/cv2 work releases the GIL anyway.
    with ThreadPoolExecutor(max_workers=_S3_FETCH_WORKERS) as pool:
        futures = [pool.submit(_fetch, wi) for wi in images]

        for future in as_completed(futures):
            wedding_image, image_bytes, fetch_err = future.result()
            try:
                if fetch_err:
                    logger.warning(f"Fetch failed for {wedding_image.id}: {fetch_err}")
                    processed_count += 1
                    continue

                try:
                    image = load_image_from_bytes(image_bytes)
                except Exception:
                    # Fallback to raw cv2 decode for the rare formats PIL doesn't grok.
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    logger.warning(f"Could not decode image {wedding_image.id}")
                    processed_count += 1
                    continue

                # Build face data for closed-eye detection.
                faces = [
                    {'landmarks': fv.landmarks, 'bbox': fv.bbox}
                    for fv in wedding_image.faces
                    if fv.landmarks
                ]

                issues = analyze_single_image(image, faces, threshold, calibrated=calibrated)

                if issues:
                    flagged_count += 1
                    for issue in issues:
                        db.add(ImageQualityFlag(
                            image_id=wedding_image.id,
                            issue_type=issue['issue_type'],
                            confidence=issue['confidence'],
                        ))

                wedding_image.quality_analyzed = True
                processed_count += 1

                # Periodic progress commit so the dashboard's status polling
                # sees movement even on large celebrations.
                if processed_count % 10 == 0:
                    job.processed_count = processed_count
                    job.flagged_count = flagged_count
                    db.commit()

            except Exception as e:
                logger.error(f"Error analyzing image {wedding_image.id}: {e}")
                continue

    # Complete the job
    job.processed_count = processed_count
    job.flagged_count = flagged_count
    job.status = "completed"
    job.completed_at = datetime.utcnow()
    db.commit()

    logger.info(f"Quality analysis complete for celebration {celebration_id}: {processed_count} images, {flagged_count} flagged")

    return job


# RQ Job wrapper function
def analyze_celebration_job(celebration_id: str, threshold: float = 0.70, reanalyze: bool = False):
    """
    RQ job wrapper for analyze_celebration.
    Creates its own database session for the background job.
    """
    from db import SessionLocal

    db = SessionLocal()
    try:
        return analyze_celebration(
            db=db,
            celebration_id=uuid.UUID(celebration_id),
            threshold=threshold,
            reanalyze=reanalyze
        )
    finally:
        db.close()
