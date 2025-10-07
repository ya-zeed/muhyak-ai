import os, logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from db import engine
from models import Base
from routers import health, uploads, search, images, cluster, celebrations

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Wedding Face Search API",
    description="Advanced face recognition and search API for wedding photos",
    version="1.0.0",
    swagger_ui_parameters={"syntaxHighlight": False},
)

# Static (optional local preview)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# CORS (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(uploads.router)
app.include_router(search.router)
app.include_router(images.router)
# app.include_router(cluster.router)
app.include_router(celebrations.router)

@app.on_event("startup")
def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logging.getLogger(__name__).info("DB ready")
    except Exception:
        logging.getLogger(__name__).exception("DB init failed")
