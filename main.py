"""
main.py

Phase 5, Step 3: FastAPI backend wrapping the PatchCore inference
service.

Endpoints:
    GET  /          -- service info (handy when someone opens the bare URL)
    GET  /health    -- liveness check
    POST /predict   -- accepts an uploaded image, returns the
                       classification, the calibrated threshold, and the
                       heatmap as base64 PNGs

Configuration (all optional; environment variables):
    ALLOWED_ORIGINS       comma-separated browser origins allowed to call
                          this API. Default: the local Vite dev server.
                          Set it to your deployed frontend URL in
                          production, e.g. https://my-app.vercel.app
    ALLOWED_ORIGIN_REGEX  optional regex for origins that change, e.g.
                          Vercel preview deployments
    MAX_UPLOAD_MB         largest accepted upload (default 10)

Run (development):
    uvicorn main:app --reload --port 8000
"""

import asyncio
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from inference import get_service

logger = logging.getLogger("uvicorn.error")  # shows up in uvicorn's console output


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    # tolerate pasted URLs with a trailing slash -- origins never have one
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]  # Vite dev server
ALLOWED_ORIGINS = _env_list("ALLOWED_ORIGINS", DEFAULT_ORIGINS)
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX") or None

MAX_UPLOAD_MB = _env_float("MAX_UPLOAD_MB", 10)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # form framing around the file itself
MAX_IMAGE_PIXELS = 25_000_000  # 25 MP; MVTec images are ~1 MP, phone photos ~12 MP


# ---------------------------------------------------------------------------
# response shape (also drives the auto-generated /docs page)
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    label: Literal["good", "defect"]
    score: float
    threshold: float
    category: str
    heatmap_base64: Optional[str] = None  # heatmap alone, for the opacity slider
    overlay_base64: str  # original with the heatmap pre-blended
    server_ms: int  # server-side time: decode + inference (+ any queue wait)


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ONE worker thread does all the model work, for two reasons:
    #  1. PatchCoreExtractor stores hook output in a shared dict, so two
    #     predictions running at once would corrupt each other.
    #  2. It keeps every torch call (including the model load) on the same
    #     thread, which is the safest setup for the MPS backend.
    # Running it in an executor (instead of directly in the async handler)
    # keeps the event loop free, so /health still answers mid-prediction.
    app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(app.state.executor, get_service)  # load once, at startup

    if "*" in ALLOWED_ORIGINS:
        logger.warning("CORS: allowing ALL origins -- fine locally, restrict before deploying")
    else:
        logger.info("CORS: allowed origins = %s", ALLOWED_ORIGINS)
    yield
    app.state.executor.shutdown(wait=False)


app = FastAPI(title="AI-Powered Quality Inspection API", lifespan=lifespan)


# Registered BEFORE the CORS middleware on purpose: the last middleware added
# is the outermost, so CORS wraps this one and a 413 still carries CORS
# headers -- otherwise the browser would hide the error behind a generic
# network failure.
@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    # Cheap early rejection from the declared size. Best effort only (a
    # client can omit or misstate it), so /predict re-checks what it reads.
    if request.method == "POST":
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"File is too large (max {MAX_UPLOAD_MB:g} MB)."},
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "AI-Powered Quality Inspection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart field: file)",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _decode_and_predict(contents: bytes) -> dict:
    """Runs on the single inference thread. Turns bad input into clean 4xx
    errors instead of letting it surface as a 500."""
    try:
        image = Image.open(io.BytesIO(contents))  # lazy: reads the header only
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=f"Image dimensions are too large (max {MAX_IMAGE_PIXELS // 1_000_000} megapixels).",
            )
        image.load()  # decode now, so a truncated file fails here, not mid-inference
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    try:
        return get_service().predict(image)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed on this image.")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large (max {MAX_UPLOAD_MB:g} MB).")
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(request.app.state.executor, _decode_and_predict, contents)
    result["server_ms"] = round((time.perf_counter() - started) * 1000)

    logger.info(
        "predict: label=%s score=%.5f upload=%.0fKB server_ms=%d",
        result["label"], result["score"], len(contents) / 1024, result["server_ms"],
    )
    return result