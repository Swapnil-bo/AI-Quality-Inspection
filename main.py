"""
main.py

Phase 5, Step 3: FastAPI backend wrapping the PatchCore inference
service. Loads the model once at startup (via inference.get_service()),
not per-request.

Endpoints:
    GET  /health   -- basic liveness check
    POST /predict  -- accepts an uploaded image, returns the
                       classification + calibrated threshold + a
                       base64 heatmap overlay

Run (development):
    uvicorn main:app --reload --port 8000
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import get_service

app = FastAPI(title="AI-Powered Quality Inspection API")

# Allow the local Vite dev server (and later, the deployed frontend)
# to call this API from the browser. Tighten allow_origins before
# deploying publicly -- "*" is fine for local development only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model():
    # Forces the PatchCore memory bank + extractor to load once, at
    # server startup, rather than lazily on the first request -- so
    # the first real prediction isn't slowed down by model loading.
    get_service()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    service = get_service()
    result = service.predict(pil_image)
    return result