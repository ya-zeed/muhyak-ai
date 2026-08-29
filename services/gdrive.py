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

FOLDER_MIME = "application/vnd.google-apps.folder"

# Drive lets one folder sit under several parents, so the "tree" can contain
# cycles, and a shared drive can nest far deeper than whoever shared it thinks.
# Walk with an explicit budget instead of trusting the shape of someone's Drive.
MAX_DEPTH = 10
MAX_FOLDERS = 500


def _list_children(folder_id: str, api_key: str) -> list[dict[str, Any]]:
    """Every image and subfolder directly inside `folder_id` (all pages).

    Asks for both kinds in one query so a folder costs one page-walk rather
    than two. `trashed = false` because Drive's file list includes the owner's
    trash by default, and photos they deleted are photos they didn't want.
    """
    children: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        params = {
            "q": (
                f"'{folder_id}' in parents and trashed = false and "
                f"(mimeType contains 'image/' or mimeType = '{FOLDER_MIME}')"
            ),
            "key": api_key,
            "fields": "nextPageToken,files(id,name,mimeType)",
            "pageSize": "1000",
        }
        if page_token:
            params["pageToken"] = page_token

        url = f"{DRIVE_API_BASE}/files?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        children.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return children


def list_folder_images(folder_id: str, api_key: str) -> list[dict[str, Any]]:
    """Return every image (id, name, mimeType) under a public Drive folder,
    descending into subfolders.

    Photographers don't hand over a flat pile of JPEGs. A typical link points
    at a folder holding one folder per photographer, each split again by day or
    by camera — so listing only the top level used to find zero images and the
    import silently did nothing.

    Breadth-first on purpose: images nearest the shared folder are queued first,
    so the gallery starts filling with the obvious ones while the deeper folders
    are still being walked.

    Duplicate filenames across subfolders are fine downstream — S3 keys carry a
    UUID and the worker de-dupes on content hash, so two different DSC_0001.jpg
    both land and the same photo filed twice lands once.
    """
    images: list[dict[str, Any]] = []
    seen_folders: set[str] = {folder_id}
    queue: list[tuple[str, int]] = [(folder_id, 0)]

    while queue:
        current, depth = queue.pop(0)
        children = _list_children(current, api_key)

        for child in children:
            if child.get("mimeType") != FOLDER_MIME:
                images.append(child)
                continue

            # Depth/'count' caps are silent by design at the API layer, but a
            # truncated import looks identical to a complete one, so say so.
            child_id = child.get("id")
            if not child_id or child_id in seen_folders:
                continue
            if depth + 1 > MAX_DEPTH:
                logger.warning(
                    "Drive walk stopped at depth %d, skipping folder %s",
                    MAX_DEPTH,
                    child.get("name"),
                )
                continue
            if len(seen_folders) >= MAX_FOLDERS:
                logger.warning(
                    "Drive walk hit the %d-folder cap; some folders were skipped",
                    MAX_FOLDERS,
                )
                continue

            seen_folders.add(child_id)
            queue.append((child_id, depth + 1))

    return images


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
