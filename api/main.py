"""FastAPI entrypoint."""
from __future__ import annotations

import asyncio
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response

ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
load_dotenv()

from .models import SearchRequest, SearchResponse  # noqa: E402
from .search import LLM_MODEL, SearchEngine  # noqa: E402

CONTACTS_DIR = Path(__file__).resolve().parent.parent / "indexer" / "data" / "contacts"
SAFE_KEY = re.compile(r"^[a-f0-9]{16}$")

engine: SearchEngine | None = None
engine_error: str | None = None
_warmup_task: asyncio.Task | None = None


async def _warmup(eng: SearchEngine) -> None:
    """Pre-touch the embedder and the LLM so the first real query doesn't
    pay the model cold-load cost. Failures are irrelevant — the request
    path degrades identically."""
    try:
        await asyncio.to_thread(eng.embedder.embed_one, "warmup")
    except Exception:
        pass
    try:
        from .search import _LLM_EXTRA_BODY
        await eng._openai.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=5,
            extra_body=_LLM_EXTRA_BODY,
            messages=[{"role": "user", "content": "ok"}],
        )
    except Exception:
        pass


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Boot degraded when the index is missing so /health can report why
    # instead of uvicorn aborting on startup.
    global engine, engine_error, _warmup_task
    try:
        engine = SearchEngine()
    except Exception as e:
        engine_error = str(e)
        print(f"[startup] engine init failed: {e}", file=sys.stderr)
    if engine is not None:
        _warmup_task = asyncio.create_task(_warmup(engine))
    yield


app = FastAPI(title="Semantic Memory Search", lifespan=_lifespan)


def _engine_unavailable_reason() -> str:
    return engine_error or "Search engine not initialized"


def _reject_browser(request: Request) -> None:
    # Browsers stamp Sec-Fetch-Site on every request (including <img> loads,
    # which CORS does not gate); the native URLSession client never does.
    # Blocks web pages from enumerating /attachment/{id} image bytes.
    if request.headers.get("sec-fetch-site"):
        raise HTTPException(403, "browser access not allowed")


@app.get("/health")
def health() -> dict:
    if engine is None:
        return {"ok": False, "error": _engine_unavailable_reason()}
    return {"ok": True}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    if engine is None:
        raise HTTPException(503, _engine_unavailable_reason())
    return await engine.search(req)


@app.get("/attachment/{att_id}")
def attachment(att_id: int, request: Request) -> Response:
    """Serve an iMessage image attachment by its rowid."""
    _reject_browser(request)
    if engine is None:
        raise HTTPException(503, _engine_unavailable_reason())
    path = engine.get_image_path(att_id)
    if not path:
        raise HTTPException(404, "not found")
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "file missing")
    data = p.read_bytes()
    media = "application/octet-stream"
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        media = "image/jpeg"
    elif suffix == ".png":
        media = "image/png"
    elif suffix in (".heic", ".heif"):
        # Many browsers/AppKit can't render HEIC directly; convert to JPEG on the fly.
        try:
            import io
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            img = Image.open(p).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            media = "image/jpeg"
        except Exception:
            media = "image/heic"
    elif suffix == ".gif":
        media = "image/gif"
    elif suffix == ".tiff":
        media = "image/tiff"
    return Response(content=data, media_type=media,
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/contact-photo/{key}")
def contact_photo(key: str, request: Request) -> Response:
    """Serve a contact photo blob by its 16-hex key. Falls back to 404 if missing."""
    _reject_browser(request)
    if not SAFE_KEY.match(key):
        raise HTTPException(400, "invalid key")
    path = CONTACTS_DIR / f"{key}.bin"
    if not path.exists():
        raise HTTPException(404, "not found")
    data = path.read_bytes()
    media = "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        media = "image/png"
    elif data[:4] == b"\xff\xd8\xff\xe0" or data[:4] == b"\xff\xd8\xff\xe1":
        media = "image/jpeg"
    return Response(content=data, media_type=media, headers={"Cache-Control": "public, max-age=86400"})
