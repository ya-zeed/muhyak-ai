"""Google Photos shared-album helpers for the background import worker.

Google offers no API for reading someone's shared album by link (the Library
API only sees app-created media since 2025), so this reads the public album
page the way a browser does:

* The share link redirects to photos.google.com/share/<album_id>?key=<key>.
* The page embeds the first ~300 items in AF_initDataCallback 'ds:1':
  [_, items, next_page_token, album_meta, ...], each item being
  [media_key, [base_url, width, height, ...], ..., meta_map].
* Further pages come from the batchexecute RPC "snAcKc".
* `<base_url>=w<W>-h<H>` serves a full-size JPEG (HEIC originals included).

Unofficial and may break if Google changes the page; listing failures raise
so the caller reports "couldn't read the album" instead of importing nothing.
Uses urllib (stdlib) so it works unchanged on Modal containers.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BATCH_URL = "https://photos.google.com/_/PhotosUi/data/batchexecute"

# Present in an item's meta map for videos; their image URL is only a frame.
VIDEO_META_KEY = "76647426"

# Guards against a token loop if Google's paging ever misbehaves.
MAX_PAGES = 50

_INIT_DATA_RE = re.compile(
    r"AF_initDataCallback\(\{key: 'ds:1'.*?data:(.*?), sideChannel", re.S
)
_SHARE_PATH_RE = re.compile(r"^/share/([A-Za-z0-9_-]+)")


def is_album_url(url: str) -> bool:
    """photos.app.goo.gl short links and photos.google.com/share links."""
    try:
        u = urllib.parse.urlparse(url.strip())
    except ValueError:
        return False
    if u.scheme != "https":
        return False
    if u.netloc == "photos.app.goo.gl":
        return len(u.path) > 1
    return u.netloc == "photos.google.com" and u.path.startswith("/share/")


def _to_images(items: Any) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    if not isinstance(items, list):
        return images
    for item in items:
        if not isinstance(item, list) or len(item) < 2:
            continue
        media_key, media = item[0], item[1]
        meta = item[-1]
        if not isinstance(media_key, str) or not isinstance(media, list):
            continue
        if isinstance(meta, dict) and VIDEO_META_KEY in meta:
            continue
        if len(media) < 3:
            continue
        base_url, width, height = media[0], media[1], media[2]
        if not isinstance(base_url, str) or not width or not height:
            continue
        images.append({
            "url": f"{base_url}=w{width}-h{height}",
            # Stable per photo, so re-running the import skips what's done.
            "name": f"gp-{media_key[-16:]}.jpg",
        })
    return images


def _fetch_page(album_id: str, key: str, token: str) -> tuple[Any, str | None]:
    inner = json.dumps([album_id, token, None, key])
    body = urllib.parse.urlencode(
        {"f.req": json.dumps([[["snAcKc", inner, None, "generic"]]])}
    ).encode()
    req = urllib.request.Request(
        BATCH_URL,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    # Response: ")]}'" guard, blank line, then a JSON envelope.
    envelope = json.loads(text[text.index("["):])
    entry = next(
        (e for e in envelope if e and e[0] == "wrb.fr" and e[1] == "snAcKc"),
        None,
    )
    if not entry or not isinstance(entry[2], str):
        raise RuntimeError("Unexpected Google Photos page response")
    data = json.loads(entry[2])
    token = data[2] if len(data) > 2 and isinstance(data[2], str) and data[2] else None
    return data[1], token


def list_album_images(album_url: str) -> list[dict[str, str]]:
    """Return every photo (url, name) in a public shared album, videos skipped.

    Raises when the album can't be read (private, deleted, or page changed).
    """
    req = urllib.request.Request(
        album_url.strip(),
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        final_url = resp.geturl()
        html = resp.read().decode("utf-8")

    parsed = urllib.parse.urlparse(final_url)
    share = _SHARE_PATH_RE.match(parsed.path)
    key = urllib.parse.parse_qs(parsed.query).get("key", [None])[0]
    match = _INIT_DATA_RE.search(html)
    if not share or not key or not match:
        raise RuntimeError("Not a readable shared album")

    data = json.loads(match.group(1))
    album_id = share.group(1)

    seen: set[str] = set()
    images: list[dict[str, str]] = []

    def add(batch: list[dict[str, str]]) -> None:
        for img in batch:
            if img["url"] not in seen:
                seen.add(img["url"])
                images.append(img)

    add(_to_images(data[1]))
    token = data[2] if isinstance(data[2], str) and data[2] else None

    pages = 0
    while token and pages < MAX_PAGES:
        items, token = _fetch_page(album_id, key, token)
        add(_to_images(items))
        pages += 1
    if token:
        logger.warning("Google Photos walk hit the %d-page cap", MAX_PAGES)

    return images


def download_photo(url: str, timeout: int = 120, attempts: int = 5) -> bytes:
    """Download one full-size photo, auto-retrying transient failures."""
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "image/jpeg,image/*;q=0.8"}
    )
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if data:
                return data
        except Exception as e:  # noqa: BLE001 — retry any transient error
            last_err = e
            logger.warning("Google Photos download attempt %d failed: %s", attempt + 1, e)
        if attempt < attempts - 1:
            time.sleep(min(2 ** (attempt + 1), 20))  # 2,4,8,16s
    if last_err:
        raise last_err
    raise RuntimeError("empty download from Google Photos")
