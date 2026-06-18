"""
ClarityNet Inference API

Endpoints:
  POST /predict          — upload image file → blur/focus prediction
  POST /predict/batch    — upload multiple images → list of predictions
  GET  /health           — liveness check
  GET  /model/info       — model metadata
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clarity.config import Config
from api.predictor import Predictor


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    cfg = Config.from_yaml("configs/default.yaml")
    predictor = Predictor(cfg)
    yield
    predictor = None


app = FastAPI(
    title="ClarityNet — Blur / Focus Detection API",
    version="0.1.0",
    description="Production CNN endpoint for detecting whether an image is in focus or blurry.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    is_sharp: bool
    label: str
    confidence: float
    sharp_probability: float
    blurry_probability: float
    blur_score: float
    blur_type: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/model/info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    cfg = predictor.cfg
    return {
        "backbone": cfg.model.backbone,
        "image_size": cfg.dataset.image_size,
        "freq_branch": cfg.model.freq_branch,
        "device": str(predictor.device),
        "checkpoint": cfg.inference.checkpoint,
        "classes": {0: "sharp", 1: "blurry"},
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP)")]):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected image file, got: {file.content_type}",
        )

    t0 = time.perf_counter()
    image_bytes = await file.read()
    try:
        result = predictor.predict_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: Annotated[list[UploadFile], File(description="Multiple image files")],
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 images per batch")

    t0 = time.perf_counter()
    predictions = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=422, detail=f"File {file.filename} is not an image")
        t1 = time.perf_counter()
        image_bytes = await file.read()
        result = predictor.predict_bytes(image_bytes)
        result["latency_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        predictions.append(result)

    return {
        "predictions": predictions,
        "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    cfg = Config.from_yaml("configs/default.yaml")
    uvicorn.run("api.main:app", host=cfg.inference.api_host, port=cfg.inference.api_port, reload=False)
