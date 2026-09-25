"""RQ worker for importing a single Google Photos image (local/dev backend).

Mirrors modal_worker.import_gphotos_image; storage and face detection are
shared with the Drive import.
"""
from jobs.gdrive_import import import_image_bytes
from services.gphotos import download_photo


def import_gphotos_image_job(
    url: str,
    filename: str,
    celebrant: str,
    photographer: str,
    celebration_id: str,
) -> None:
    import_image_bytes(
        lambda: download_photo(url),
        filename,
        "image/jpeg",
        celebrant,
        photographer,
        celebration_id,
        progress_prefix="gphotos_import",
    )
