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


# T009: Closed eyes detection using face landmarks
def detect_closed_eyes(image: np.ndarray, faces: list, ear_threshold: float = 0.2) -> tuple[bool, float]:
    """
    Detect closed eyes using Eye Aspect Ratio (EAR) from facial landmarks.

    Args:
        image: BGR image array from OpenCV
        faces: List of face detection results with landmarks
        ear_threshold: EAR threshold below which eyes are considered closed

    Returns:
        tuple of (has_closed_eyes, confidence_score)
    """
    if not faces:
        return False, 0.0

    max_closed_confidence = 0.0
    has_any_closed = False

    for face in faces:
        landmarks = face.get('landmarks') or face.get('kps')
        if landmarks is None:
            continue

        landmarks = np.array(landmarks)
        if len(landmarks) < 5:
            continue

        # InsightFace 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        # For closed eye detection, we need to analyze eye region
        # Since we only have center points, we'll use a heuristic based on eye region intensity

        left_eye = landmarks[0] if len(landmarks) > 0 else None
        right_eye = landmarks[1] if len(landmarks) > 1 else None

        if left_eye is None or right_eye is None:
            continue

        # Sample eye regions for darkness (closed eyes tend to be darker/narrower)
        # This is a simplified heuristic without full facial landmark mesh
        eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
        sample_radius = int(eye_distance * 0.15)

        if sample_radius < 3:
            sample_radius = 3

        h, w = image.shape[:2]

        def get_eye_intensity(eye_point):
            x, y = int(eye_point[0]), int(eye_point[1])
            x = max(sample_radius, min(w - sample_radius, x))
            y = max(sample_radius, min(h - sample_radius, y))
            region = image[y-sample_radius:y+sample_radius, x-sample_radius:x+sample_radius]
            if region.size == 0:
                return 128
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            return gray_region.mean()

        left_intensity = get_eye_intensity(left_eye)
        right_intensity = get_eye_intensity(right_eye)
        avg_intensity = (left_intensity + right_intensity) / 2

        # Closed eyes tend to have lower intensity (eyelid skin vs eyeball)
        # This is a heuristic - production would use a dedicated eye state classifier
        is_likely_closed = avg_intensity < 80  # Dark eye regions

        if is_likely_closed:
            has_any_closed = True
            confidence = 1.0 - (avg_intensity / 80)
            max_closed_confidence = max(max_closed_confidence, confidence)

    return has_any_closed, max_closed_confidence


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
    threshold: float = 0.70
) -> list[dict]:
    """
    Analyze a single image for all quality issues.

    Args:
        image: BGR image array from OpenCV
        faces: Optional list of detected faces with landmarks
        threshold: Minimum confidence threshold for flagging issues

    Returns:
        List of detected issues with type and confidence
    """
    issues = []

    # Run all detectors
    is_blurry, blur_conf = detect_blur(image)
    if is_blurry and blur_conf >= threshold:
        issues.append({"issue_type": "blur", "confidence": blur_conf})

    has_motion_blur, motion_conf = detect_motion_blur(image)
    if has_motion_blur and motion_conf >= threshold:
        issues.append({"issue_type": "motion_blur", "confidence": motion_conf})

    if faces:
        has_closed_eyes, eyes_conf = detect_closed_eyes(image, faces)
        if has_closed_eyes and eyes_conf >= threshold:
            issues.append({"issue_type": "closed_eyes", "confidence": eyes_conf})

    is_underexposed, under_conf = detect_underexposed(image)
    if is_underexposed and under_conf >= threshold:
        issues.append({"issue_type": "underexposed", "confidence": under_conf})

    is_overexposed, over_conf = detect_overexposed(image)
    if is_overexposed and over_conf >= threshold:
        issues.append({"issue_type": "overexposed", "confidence": over_conf})

    return issues


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
    from config import settings
    import boto3
    from io import BytesIO

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

    flagged_count = 0
    processed_count = 0

    for wedding_image in images:
        try:
            # Download image from S3
            # Extract bucket and key from file_path URL
            file_path = wedding_image.compressed_file_path or wedding_image.file_path

            # Parse S3 URL to get key
            # URL format: https://bucket.endpoint/key or https://endpoint/bucket/key
            if settings.AWS_S3_BUCKET in file_path:
                key = file_path.split(settings.AWS_S3_BUCKET + '/')[-1].split('?')[0]
            else:
                key = '/'.join(file_path.split('/')[-2:]).split('?')[0]

            response = s3_client.get_object(Bucket=settings.AWS_S3_BUCKET, Key=key)
            image_bytes = response['Body'].read()

            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.warning(f"Could not decode image {wedding_image.id}")
                processed_count += 1
                continue

            # Get face data for closed eye detection
            faces = []
            for face_vector in wedding_image.faces:
                if face_vector.landmarks:
                    faces.append({
                        'landmarks': face_vector.landmarks,
                        'bbox': face_vector.bbox
                    })

            # Analyze image
            issues = analyze_single_image(image, faces, threshold)

            # Store flags
            if issues:
                flagged_count += 1
                for issue in issues:
                    flag = ImageQualityFlag(
                        image_id=wedding_image.id,
                        issue_type=issue['issue_type'],
                        confidence=issue['confidence']
                    )
                    db.add(flag)

            # Mark as analyzed
            wedding_image.quality_analyzed = True
            processed_count += 1

            # Update job progress periodically
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
