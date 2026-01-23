"""
Quality Analysis Router - API endpoints for image quality detection

T015-T020: Endpoints for quality analysis triggering, status, flags, and background jobs
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from db import get_db
from models import (
    Celebration,
    WeddingImage,
    ImageQualityFlag,
    QualityAnalysisJob,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quality", tags=["Quality Analysis"])


# Pydantic models for request/response
class TriggerAnalysisRequest(BaseModel):
    threshold: float = 0.70
    reanalyze: bool = False


class QualityAnalysisJobResponse(BaseModel):
    id: uuid.UUID
    celebration_id: uuid.UUID
    total_images: int
    processed_count: int
    flagged_count: int
    status: str
    threshold: float
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class QualityFlagResponse(BaseModel):
    id: uuid.UUID
    issue_type: str
    confidence: float
    reviewed: bool
    dismissed: bool
    created_at: datetime

    class Config:
        from_attributes = True


class FlaggedImageResponse(BaseModel):
    image_id: uuid.UUID
    filename: str
    file_path: str
    compressed_file_path: str
    flags: list[QualityFlagResponse]
    all_reviewed: bool
    all_dismissed: bool


class FlaggedImagesListResponse(BaseModel):
    items: list[FlaggedImageResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


class QualitySummaryResponse(BaseModel):
    total_images: int
    analyzed_images: int
    flagged_images: int
    reviewed_images: int
    dismissed_images: int
    issues_by_type: dict[str, int]
    last_analysis: Optional[QualityAnalysisJobResponse]


class UpdateFlagRequest(BaseModel):
    reviewed: Optional[bool] = None
    dismissed: Optional[bool] = None


class BulkUpdateFlagRequest(BaseModel):
    image_ids: list[uuid.UUID]
    reviewed: Optional[bool] = None
    dismissed: Optional[bool] = None


class BulkUpdateResponse(BaseModel):
    updated_count: int


def _get_celebration(db: Session, photographer: str, celebrant: str) -> Celebration:
    """Helper to get celebration by photographer/celebrant slugs"""
    celebration = db.query(Celebration).filter(
        Celebration.photographer == photographer,
        Celebration.celebrant == celebrant
    ).first()

    if not celebration:
        raise HTTPException(status_code=404, detail="Celebration not found")

    return celebration


# T015: POST /quality/{photographer}/{celebrant}/analyze
@router.post("/{photographer}/{celebrant}/analyze", response_model=QualityAnalysisJobResponse, status_code=202)
def trigger_quality_analysis(
    photographer: str,
    celebrant: str,
    request: TriggerAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Trigger quality analysis for a celebration.
    Returns immediately with job ID; analysis runs in background.
    """
    celebration = _get_celebration(db, photographer, celebrant)

    # Check if analysis already in progress
    existing_job = db.query(QualityAnalysisJob).filter(
        QualityAnalysisJob.celebration_id == celebration.id,
        QualityAnalysisJob.status.in_(["pending", "processing"])
    ).first()

    if existing_job:
        raise HTTPException(status_code=409, detail="Analysis already in progress for this celebration")

    # Count images to analyze
    query = db.query(WeddingImage).filter(WeddingImage.celebration_id == celebration.id)
    if not request.reanalyze:
        query = query.filter(WeddingImage.quality_analyzed == False)

    total_images = query.count()

    # Create job record
    job = QualityAnalysisJob(
        celebration_id=celebration.id,
        total_images=total_images,
        processed_count=0,
        flagged_count=0,
        status="pending",
        threshold=request.threshold
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # T020: Enqueue background job
    enqueue_quality_analysis(str(celebration.id), request.threshold, request.reanalyze)

    logger.info(f"Quality analysis triggered for celebration {celebration.id}, job {job.id}")

    return job


# T016: GET /quality/{photographer}/{celebrant}/status
@router.get("/{photographer}/{celebrant}/status", response_model=QualityAnalysisJobResponse)
def get_quality_status(
    photographer: str,
    celebrant: str,
    db: Session = Depends(get_db)
):
    """Get the most recent quality analysis job status for a celebration."""
    celebration = _get_celebration(db, photographer, celebrant)

    job = db.query(QualityAnalysisJob).filter(
        QualityAnalysisJob.celebration_id == celebration.id
    ).order_by(QualityAnalysisJob.started_at.desc()).first()

    if not job:
        raise HTTPException(status_code=404, detail="No analysis job found for this celebration")

    return job


# T017: GET /quality/{photographer}/{celebrant}/flags
@router.get("/{photographer}/{celebrant}/flags", response_model=FlaggedImagesListResponse)
def list_flagged_images(
    photographer: str,
    celebrant: str,
    issue_type: Optional[str] = Query(None, description="Filter by issue type"),
    reviewed: Optional[bool] = Query(None, description="Filter by reviewed status"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List images with quality flags for a celebration."""
    celebration = _get_celebration(db, photographer, celebrant)

    # Build query for images with flags
    base_query = db.query(WeddingImage).join(
        ImageQualityFlag, WeddingImage.id == ImageQualityFlag.image_id
    ).filter(
        WeddingImage.celebration_id == celebration.id
    )

    if issue_type:
        base_query = base_query.filter(ImageQualityFlag.issue_type == issue_type)

    if reviewed is not None:
        base_query = base_query.filter(ImageQualityFlag.reviewed == reviewed)

    # Get distinct images
    base_query = base_query.distinct()

    # Count total
    total = base_query.count()
    total_pages = (total + per_page - 1) // per_page

    # Paginate
    images = base_query.offset((page - 1) * per_page).limit(per_page).all()

    # Build response
    items = []
    for image in images:
        flags = [
            QualityFlagResponse(
                id=flag.id,
                issue_type=flag.issue_type,
                confidence=flag.confidence,
                reviewed=flag.reviewed,
                dismissed=flag.dismissed,
                created_at=flag.created_at
            )
            for flag in image.quality_flags
            if (issue_type is None or flag.issue_type == issue_type)
            and (reviewed is None or flag.reviewed == reviewed)
        ]

        all_reviewed = all(f.reviewed for f in image.quality_flags)
        all_dismissed = all(f.dismissed for f in image.quality_flags)

        items.append(FlaggedImageResponse(
            image_id=image.id,
            filename=image.filename,
            file_path=image.file_path,
            compressed_file_path=image.compressed_file_path,
            flags=flags,
            all_reviewed=all_reviewed,
            all_dismissed=all_dismissed
        ))

    return FlaggedImagesListResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages
    )


# T018: GET /quality/{photographer}/{celebrant}/summary
@router.get("/{photographer}/{celebrant}/summary", response_model=QualitySummaryResponse)
def get_quality_summary(
    photographer: str,
    celebrant: str,
    db: Session = Depends(get_db)
):
    """Get quality analysis summary statistics for a celebration."""
    celebration = _get_celebration(db, photographer, celebrant)

    # Total images
    total_images = db.query(WeddingImage).filter(
        WeddingImage.celebration_id == celebration.id
    ).count()

    # Analyzed images
    analyzed_images = db.query(WeddingImage).filter(
        WeddingImage.celebration_id == celebration.id,
        WeddingImage.quality_analyzed == True
    ).count()

    # Flagged images (distinct images with at least one flag)
    flagged_images = db.query(func.count(func.distinct(ImageQualityFlag.image_id))).join(
        WeddingImage, ImageQualityFlag.image_id == WeddingImage.id
    ).filter(
        WeddingImage.celebration_id == celebration.id
    ).scalar() or 0

    # Reviewed images (all flags reviewed)
    reviewed_subq = db.query(ImageQualityFlag.image_id).join(
        WeddingImage, ImageQualityFlag.image_id == WeddingImage.id
    ).filter(
        WeddingImage.celebration_id == celebration.id
    ).group_by(ImageQualityFlag.image_id).having(
        func.bool_and(ImageQualityFlag.reviewed) == True
    ).subquery()

    reviewed_images = db.query(func.count()).select_from(reviewed_subq).scalar() or 0

    # Dismissed images
    dismissed_subq = db.query(ImageQualityFlag.image_id).join(
        WeddingImage, ImageQualityFlag.image_id == WeddingImage.id
    ).filter(
        WeddingImage.celebration_id == celebration.id
    ).group_by(ImageQualityFlag.image_id).having(
        func.bool_and(ImageQualityFlag.dismissed) == True
    ).subquery()

    dismissed_images = db.query(func.count()).select_from(dismissed_subq).scalar() or 0

    # Issues by type
    issues_by_type_query = db.query(
        ImageQualityFlag.issue_type,
        func.count(ImageQualityFlag.id)
    ).join(
        WeddingImage, ImageQualityFlag.image_id == WeddingImage.id
    ).filter(
        WeddingImage.celebration_id == celebration.id
    ).group_by(ImageQualityFlag.issue_type).all()

    issues_by_type = {row[0]: row[1] for row in issues_by_type_query}

    # Last analysis job
    last_job = db.query(QualityAnalysisJob).filter(
        QualityAnalysisJob.celebration_id == celebration.id
    ).order_by(QualityAnalysisJob.started_at.desc()).first()

    return QualitySummaryResponse(
        total_images=total_images,
        analyzed_images=analyzed_images,
        flagged_images=flagged_images,
        reviewed_images=reviewed_images,
        dismissed_images=dismissed_images,
        issues_by_type=issues_by_type,
        last_analysis=QualityAnalysisJobResponse.model_validate(last_job) if last_job else None
    )


# PATCH endpoints for US2 (T030, T031) - keeping them here for router completeness
@router.patch("/{photographer}/{celebrant}/flags/{image_id}", response_model=FlaggedImageResponse)
def update_quality_flag(
    photographer: str,
    celebrant: str,
    image_id: uuid.UUID,
    request: UpdateFlagRequest,
    db: Session = Depends(get_db)
):
    """Update reviewed/dismissed status for all flags on an image."""
    celebration = _get_celebration(db, photographer, celebrant)

    image = db.query(WeddingImage).filter(
        WeddingImage.id == image_id,
        WeddingImage.celebration_id == celebration.id
    ).first()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Update all flags for this image
    for flag in image.quality_flags:
        if request.reviewed is not None:
            flag.reviewed = request.reviewed
        if request.dismissed is not None:
            flag.dismissed = request.dismissed

    db.commit()

    flags = [
        QualityFlagResponse(
            id=flag.id,
            issue_type=flag.issue_type,
            confidence=flag.confidence,
            reviewed=flag.reviewed,
            dismissed=flag.dismissed,
            created_at=flag.created_at
        )
        for flag in image.quality_flags
    ]

    return FlaggedImageResponse(
        image_id=image.id,
        filename=image.filename,
        file_path=image.file_path,
        compressed_file_path=image.compressed_file_path,
        flags=flags,
        all_reviewed=all(f.reviewed for f in image.quality_flags),
        all_dismissed=all(f.dismissed for f in image.quality_flags)
    )


@router.patch("/{photographer}/{celebrant}/flags/bulk", response_model=BulkUpdateResponse)
def bulk_update_quality_flags(
    photographer: str,
    celebrant: str,
    request: BulkUpdateFlagRequest,
    db: Session = Depends(get_db)
):
    """Bulk update reviewed/dismissed status for multiple images."""
    celebration = _get_celebration(db, photographer, celebrant)

    if len(request.image_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per bulk request")

    # Update flags for all specified images
    updated_count = 0
    for image_id in request.image_ids:
        flags = db.query(ImageQualityFlag).join(
            WeddingImage, ImageQualityFlag.image_id == WeddingImage.id
        ).filter(
            WeddingImage.id == image_id,
            WeddingImage.celebration_id == celebration.id
        ).all()

        for flag in flags:
            if request.reviewed is not None:
                flag.reviewed = request.reviewed
            if request.dismissed is not None:
                flag.dismissed = request.dismissed
            updated_count += 1

    db.commit()

    return BulkUpdateResponse(updated_count=updated_count)


# T020: Background job enqueue function
def enqueue_quality_analysis(celebration_id: str, threshold: float, reanalyze: bool = False):
    """Enqueue a quality analysis job to configured backend (RQ or Modal)."""
    from jobs.dispatcher import dispatch_job

    job_id = dispatch_job(
        "quality_analysis",
        celebration_id=celebration_id,
        threshold=threshold,
        reanalyze=reanalyze,
    )

    logger.info(f"Enqueued quality analysis job {job_id} for celebration {celebration_id}")
    return job_id
