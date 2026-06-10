"""
Job Dispatcher - Routes background jobs to configured backend (RQ or Modal)

Usage:
    from jobs.dispatcher import dispatch_job

    # Dispatch a job to the configured backend
    dispatch_job("process_image", image_bytes=content, image_id=str(img_id), ...)
"""
import os
import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded connections
_rq_queue = None
_modal_functions = {}


def _get_rq_queue():
    """Get or create RQ queue connection."""
    global _rq_queue
    if _rq_queue is None:
        import redis
        from rq import Queue
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_conn = redis.from_url(redis_url)
        _rq_queue = Queue("default", connection=redis_conn)
    return _rq_queue


def _dispatch_rq(job_type: str, **kwargs) -> str:
    """Dispatch job to Redis Queue."""
    queue = _get_rq_queue()

    job_mapping = {
        "process_image": "routers.uploads._handle_single_upload",
        "quality_analysis": "services.quality_analyzer.analyze_celebration_job",
        "reprocess_image": "jobs.reprocess.reprocess_image_job",
        "import_drive_image": "jobs.gdrive_import.import_drive_image_job",
    }

    func_path = job_mapping.get(job_type)
    if not func_path:
        raise ValueError(f"Unknown job type: {job_type}")

    # Map kwargs to positional args based on job type
    if job_type == "process_image":
        job = queue.enqueue(
            func_path,
            kwargs.get("celebrant"),
            kwargs.get("photographer"),
            kwargs.get("filename"),
            kwargs.get("content"),
            kwargs.get("celebration_id"),
        )
    elif job_type == "quality_analysis":
        job = queue.enqueue(
            func_path,
            kwargs.get("celebration_id"),
            kwargs.get("threshold", 0.70),
            kwargs.get("reanalyze", False),
            job_timeout=600,
        )
    elif job_type == "reprocess_image":
        job = queue.enqueue(
            func_path,
            kwargs.get("image_id"),
        )
    elif job_type == "import_drive_image":
        job = queue.enqueue(
            func_path,
            kwargs.get("file_id"),
            kwargs.get("api_key"),
            kwargs.get("filename"),
            kwargs.get("mime_type"),
            kwargs.get("celebrant"),
            kwargs.get("photographer"),
            kwargs.get("celebration_id"),
            job_timeout=600,
        )
    else:
        raise ValueError(f"Unknown job type: {job_type}")

    logger.info(f"[RQ] Dispatched {job_type} job: {job.id}")
    return job.id


def _dispatch_modal(job_type: str, **kwargs) -> str:
    """Dispatch job to Modal serverless functions."""
    try:
        import modal
    except ImportError:
        raise RuntimeError("Modal is not installed. Run: pip install modal")

    app_name = settings.MODAL_APP_NAME

    if job_type == "process_image":
        # Call the Modal function - Modal 1.x syntax uses from_name()
        process_fn = modal.Function.from_name(app_name, "process_image")
        # spawn() returns immediately, runs in background
        call = process_fn.spawn(
            image_bytes=kwargs.get("content"),
            celebrant=kwargs.get("celebrant"),
            photographer=kwargs.get("photographer"),
            filename=kwargs.get("filename"),
            celebration_id=kwargs.get("celebration_id"),
        )
        job_id = call.object_id
        logger.info(f"[Modal] Dispatched process_image job: {job_id}")
        return job_id

    elif job_type == "quality_analysis":
        analyze_fn = modal.Function.from_name(app_name, "analyze_quality")
        call = analyze_fn.spawn(
            celebration_id=kwargs.get("celebration_id"),
            threshold=kwargs.get("threshold", 0.70),
            reanalyze=kwargs.get("reanalyze", False),
        )
        job_id = call.object_id
        logger.info(f"[Modal] Dispatched quality_analysis job: {job_id}")
        return job_id

    elif job_type == "reprocess_image":
        reprocess_fn = modal.Function.from_name(app_name, "reprocess_image")
        call = reprocess_fn.spawn(image_id=kwargs.get("image_id"))
        job_id = call.object_id
        logger.info(f"[Modal] Dispatched reprocess_image job: {job_id}")
        return job_id

    elif job_type == "import_drive_image":
        import_fn = modal.Function.from_name(app_name, "import_drive_image")
        call = import_fn.spawn(
            file_id=kwargs.get("file_id"),
            api_key=kwargs.get("api_key"),
            filename=kwargs.get("filename"),
            mime_type=kwargs.get("mime_type"),
            celebrant=kwargs.get("celebrant"),
            photographer=kwargs.get("photographer"),
            celebration_id=kwargs.get("celebration_id"),
        )
        job_id = call.object_id
        logger.info(f"[Modal] Dispatched import_drive_image job: {job_id}")
        return job_id

    else:
        raise ValueError(f"Unknown job type: {job_type}")


def dispatch_job(job_type: str, **kwargs) -> str:
    """
    Dispatch a background job to the configured backend.

    Args:
        job_type: One of "process_image", "quality_analysis", "reprocess_image"
        **kwargs: Job-specific arguments

    Returns:
        Job ID string

    Job types and their kwargs:
        process_image:
            - celebrant: str
            - photographer: str
            - filename: str
            - content: bytes
            - celebration_id: str

        quality_analysis:
            - celebration_id: str
            - threshold: float (default 0.70)
            - reanalyze: bool (default False)

        reprocess_image:
            - image_id: str
    """
    backend = settings.WORKER_BACKEND.lower()

    if backend == "rq":
        return _dispatch_rq(job_type, **kwargs)
    elif backend == "modal":
        return _dispatch_modal(job_type, **kwargs)
    else:
        raise ValueError(f"Unknown WORKER_BACKEND: {backend}. Use 'rq' or 'modal'.")


def get_backend_info() -> dict:
    """Get information about the configured backend."""
    backend = settings.WORKER_BACKEND.lower()

    if backend == "rq":
        return {
            "backend": "rq",
            "description": "Redis Queue (self-hosted workers)",
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        }
    elif backend == "modal":
        return {
            "backend": "modal",
            "description": "Modal.com (serverless, pay-per-use)",
            "app_name": settings.MODAL_APP_NAME,
        }
    else:
        return {"backend": backend, "description": "Unknown backend"}
