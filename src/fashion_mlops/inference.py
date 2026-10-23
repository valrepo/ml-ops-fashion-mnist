"""Model loading (from MinIO) and single-image prediction."""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import torch

from . import CLASS_NAMES
from .model import LeNet5
from .preprocessing import preprocess_bytes
from .storage import ModelStore

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """No model is loaded (not trained yet, or MinIO unavailable)."""


@dataclass
class LoadedModel:
    model: torch.nn.Module
    class_names: list[str]
    info: dict = field(default_factory=dict)

    @classmethod
    def from_bytes(cls, data: bytes, info: dict | None = None) -> "LoadedModel":
        ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        model = LeNet5(num_classes=ckpt.get("model_config", {}).get("num_classes", len(CLASS_NAMES)))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        meta = {k: ckpt.get(k) for k in ("run_id", "val_accuracy", "created_at")}
        return cls(model=model, class_names=list(ckpt.get("class_names", CLASS_NAMES)),
                   info={**meta, **(info or {})})

    @torch.no_grad()
    def predict_array(self, x: np.ndarray) -> dict:
        logits = self.model(torch.from_numpy(x))
        probs = torch.softmax(logits, dim=1)[0].numpy()
        idx = int(probs.argmax())
        return {
            "class_id": idx,
            "class_name": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
        }


class ModelService:
    """Thread-safe holder that lazily (re)loads the selected model from MinIO."""

    def __init__(self, store_factory, selector: str = "latest"):
        self._store_factory = store_factory
        self.selector = selector
        self._loaded: LoadedModel | None = None
        self.last_error: str | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> LoadedModel | None:
        return self._loaded

    def load(self) -> LoadedModel:
        with self._lock:
            selector = self.selector
            try:
                store: ModelStore = self._store_factory()
                key = store.resolve_key(selector)
                data = store.download(key)
                loaded = LoadedModel.from_bytes(
                    data, info={"bucket": store.bucket, "object_key": key, "selector": selector}
                )
            except Exception as exc:  # keep serving the previous model, expose the error
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Model load failed (%s): %s", selector, self.last_error)
                raise ModelUnavailableError(self.last_error) from exc
            self._loaded, self.last_error = loaded, None
            log.info("Loaded model %s/%s", loaded.info["bucket"], loaded.info["object_key"])
            return loaded

    def get(self) -> LoadedModel:
        """Return the current model, trying to load it on demand if missing."""
        return self._loaded or self.load()

    def predict(self, image_bytes: bytes, auto_invert: bool = True) -> dict:
        model = self.get()
        x = preprocess_bytes(image_bytes, auto_invert=auto_invert)
        return {**model.predict_array(x), "model": model.info}
