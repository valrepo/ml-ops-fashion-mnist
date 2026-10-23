"""FastAPI prediction service.

Endpoints
---------
GET  /health          liveness + model status
POST /predict         multipart upload ``file=<image>`` -> predicted class
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .config import StorageSettings
from .inference import ModelService, ModelUnavailableError
from .preprocessing import MAX_UPLOAD_BYTES, InvalidImageError
from .storage import ModelStore

log = logging.getLogger("fashion_mlops.api")


class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    model: dict


def default_service() -> ModelService:
    return ModelService(
        store_factory=lambda: ModelStore(StorageSettings.from_env()),
        selector=os.environ.get("MODEL_SELECTOR", "latest"),
    )


def create_app(service: ModelService | None = None, load_on_startup: bool = True) -> FastAPI:
    service = service or default_service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_on_startup:
            try:
                await run_in_threadpool(service.load)
            except ModelUnavailableError:
                log.warning("Starting without a model; it will be loaded on first request.")
        yield

    app = FastAPI(
        title="Fashion-MNIST LeNet-5 API",
        version="0.1.0",
        description="Upload a single image, get the predicted Fashion-MNIST class.",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        loaded = service.loaded
        return {
            "status": "ok",
            "model_loaded": loaded is not None,
            "model": loaded.info if loaded else None,
            "selector": service.selector,
            "error": service.last_error,
        }

    @app.post("/predict", response_model=Prediction)
    async def predict(
        file: UploadFile = File(..., description="A single image (PNG, JPEG, BMP, GIF, TIFF, WEBP)"),
        auto_invert: bool = Query(True, description="Invert light-background images to match Fashion-MNIST"),
    ) -> dict:
        if file.content_type and not (
            file.content_type.startswith("image/") or file.content_type == "application/octet-stream"
        ):
            raise HTTPException(415, detail=f"Unsupported content type {file.content_type!r}; upload an image.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            return await run_in_threadpool(service.predict, data, auto_invert)
        except InvalidImageError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            raise HTTPException(503, detail=_unavailable_msg(str(exc))) from exc

    return app


def _unavailable_msg(error: str | None) -> str:
    return (
        "Model not available. Train a model first (docker compose --profile train run --rm trainer) "
        f"and check MinIO. Details: {error or 'model not loaded'}"
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
