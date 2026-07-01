"""Google Drive helpers for the background import worker.

Lists a public folder, downloads originals, and compresses to the platform's
display size. Uses urllib (stdlib) so it works unchanged on Modal containers.
"""
from __future__ import annotations

import io
import json
import time
import logging
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

from PIL import Image, ImageOps

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"


def list_folder_images(folder_id: str, api_key: str) -> list[dict[str, Any]]:
    """Return all image files (id, name, mimeType) in a public Drive folder."""
    files: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and mimeType contains 'image/'",
            "key": api_key,
            "fields": "nextPageToken,files(id,name,mimeType)",
            "pageSize": "1000",
        }
        if page_token:
            params["pageToken"] = page_token

        url = f"{DRIVE_API_BASE}/files?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return files


def download_drive_file(
    file_id: str, api_key: str, timeout: int = 120, attempts: int = 5
) -> bytes:
    """Download a single Drive file's bytes, auto-retrying transient failures.

    Drive throttles bursts of parallel downloads (429/5xx) and connections
    time out; retry with exponential backoff so images heal themselves instead
    of failing the import.
    """
    params = urllib.parse.urlencode({"alt": "media", "key": api_key})
    url = f"{DRIVE_API_BASE}/files/{file_id}?{params}"
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            if data:
                return data
        except Exception as e:  # noqa: BLE001 — retry any transient error
            last_err = e
            logger.warning("Drive download attempt %d failed: %s", attempt + 1, e)
        if attempt < attempts - 1:
            time.sleep(min(2 ** (attempt + 1), 20))  # 2,4,8,16s
    if last_err:
        raise last_err
    raise RuntimeError("empty download from Drive")


def compress_image(raw: bytes, max_edge: int = 2048, quality: int = 72) -> bytes:
    """Downscale to `max_edge` (long side) and re-encode as JPEG. EXIF rotation
    is baked in so the stored bytes need no further orientation handling."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
