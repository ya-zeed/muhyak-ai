from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, Float, Integer, Text, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY

try:
    from pgvector.sqlalchemy import Vector
except ImportError:  # pragma: no cover - keeps imports working if pgvector is uninstalled
    Vector = None


class Base(DeclarativeBase): pass


class Celebration(Base):
    __tablename__ = "celebrations"
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    celebrant: Mapped[str] = mapped_column(String, nullable=False)
    photographer: Mapped[str] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    images: Mapped[list["WeddingImage"]] = relationship(
        back_populates="celebration",
        cascade="all, delete-orphan"
    )
    faces: Mapped[list["FaceVector"]] = relationship(
        back_populates="celebration",
        cascade="all, delete-orphan"
    )


class WeddingImage(Base):
    __tablename__ = "wedding_images"
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    celebration_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("celebrations.id"),
                                                      nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)  # original (private or signed)
    compressed_file_path: Mapped[str] = mapped_column(String, nullable=False)
    file_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    upload_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    faces_count: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[str] = mapped_column(String, default="pending")  # pending|processing|completed|failed
    extra_metadata: Mapped[str | None] = mapped_column(Text)
    order_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    quality_analyzed: Mapped[bool] = mapped_column(default=False)  # T002: Whether quality analysis has run

    celebration: Mapped["Celebration"] = relationship(back_populates="images")
    faces: Mapped[list["FaceVector"]] = relationship(back_populates="image", cascade="all, delete-orphan")
    quality_flags: Mapped[list["ImageQualityFlag"]] = relationship(back_populates="image", cascade="all, delete-orphan")



class FaceVector(Base):
    __tablename__ = "face_vectors"
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("wedding_images.id"), nullable=False)
    celebration_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("celebrations.id"),
                                                      nullable=False)
    face_index: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)  # legacy column; pgvector column added in migration 003
    vector_pg: Mapped[list[float] | None] = mapped_column(Vector(512) if Vector is not None else ARRAY(Float), nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    landmarks: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    confidence: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float)
    embedding_model: Mapped[str | None] = mapped_column(String(40), default=None, index=True)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    celebration: Mapped["Celebration"] = relationship(back_populates="faces")
    image: Mapped["WeddingImage"] = relationship(back_populates="faces")


# T003: Quality Analysis Job - tracks progress of quality analysis for a celebration
class QualityAnalysisJob(Base):
    __tablename__ = "quality_analysis_jobs"
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    celebration_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("celebrations.id"), nullable=False)
    total_images: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_count: Mapped[int] = mapped_column(Integer, default=0)
    flagged_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|processing|completed|failed
    threshold: Mapped[float] = mapped_column(Float, default=0.70)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    celebration: Mapped["Celebration"] = relationship()


# T004: Image Quality Flag - detected quality issues for an image
class ImageQualityFlag(Base):
    __tablename__ = "image_quality_flags"
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("wedding_images.id", ondelete="CASCADE"), nullable=False)
    issue_type: Mapped[str] = mapped_column(String(50), nullable=False)  # blur|motion_blur|closed_eyes|underexposed|overexposed
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reviewed: Mapped[bool] = mapped_column(default=False)
    dismissed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    image: Mapped["WeddingImage"] = relationship(back_populates="quality_flags")
