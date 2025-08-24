from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, Float, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY


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
    celebration: Mapped["Celebration"] = relationship(back_populates="images")
    faces: Mapped[list["FaceVector"]] = relationship(back_populates="image", cascade="all, delete-orphan")


class FaceVector(Base):
    __tablename__ = "face_vectors"
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("wedding_images.id"), nullable=False)
    celebration_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("celebrations.id"),
                                                      nullable=False)
    face_index: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)  # step 1: keep ARRAY; pgvector later
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    landmarks: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    confidence: Mapped[float | None] = mapped_column(Float)
    quality_score: Mapped[float | None] = mapped_column(Float)
    created_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    celebration: Mapped["Celebration"] = relationship(back_populates="faces")
    image: Mapped["WeddingImage"] = relationship(back_populates="faces")
