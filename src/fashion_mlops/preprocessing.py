"""Image preprocessing shared by training and inference.

Training images come from torchvision as 28x28 uint8 grayscale.
Uploaded images are brought into the same format before the
identical normalisation is applied.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = 28
# Fashion-MNIST training-set statistics (pixel values scaled to [0, 1]).
MEAN = 0.2860
STD = 0.3530

ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "GIF", "TIFF", "WEBP"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as a supported image."""


def load_image(data: bytes) -> Image.Image:
    """Decode bytes into a PIL image, validating format and size."""
    if not data:
        raise InvalidImageError("Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("File is not a readable image.") from exc
    if img.format not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format {img.format!r}; use one of {sorted(ALLOWED_FORMATS)}."
        )
    return img


def to_fashion_mnist_array(img: Image.Image, auto_invert: bool = True) -> np.ndarray:
    """Convert any PIL image to a 28x28 float32 array in [0, 1] (Fashion-MNIST style).
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    gray = img.convert("L")

    if auto_invert:
        arr = np.asarray(gray, dtype=np.float32)
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        if border.mean() > 127:
            gray = ImageOps.invert(gray)

    if gray.size != (IMAGE_SIZE, IMAGE_SIZE):
        side = max(gray.size)
        fill = int(np.asarray(gray)[0, 0])  # pad with the (already dark) corner colour
        gray = ImageOps.pad(gray, (side, side), color=fill)
        gray = gray.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    return np.asarray(gray, dtype=np.float32) / 255.0


def normalize(arr: np.ndarray) -> np.ndarray:
    """Apply the training normalisation to an array in [0, 1]."""
    return (arr - MEAN) / STD


def preprocess_bytes(data: bytes, auto_invert: bool = True) -> np.ndarray:
    """Bytes -> normalised float32 array of shape (1, 1, 28, 28), ready for the model."""
    arr = to_fashion_mnist_array(load_image(data), auto_invert=auto_invert)
    return normalize(arr)[None, None, :, :].astype(np.float32)
