from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    DATABASE_URL: str = Field("postgresql+psycopg://postgres:postgres@db:5432/wedding_faces")
    REDIS_URL: str = Field("redis://redis:6379")

    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_S3_BUCKET: str | None = None
    AWS_REGION: str = "nyc3"  # DO Spaces region slug works fine here

    # New: S3-compatible endpoint for Spaces
    S3_ENDPOINT: str | None = None  # e.g., https://nyc3.digitaloceanspaces.com
    # Optional public base for returned URLs (CDN/custom domain)
    PUBLIC_S3_BASE_URL: str | None = None

    UPLOAD_DIR: str = "uploads"
    VECTOR_DIM: int = 512
    INSIGHTFACE_PROVIDER: str = "CPUExecutionProvider"
    DET_SIZE_W: int = 640
    DET_SIZE_H: int = 640

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
