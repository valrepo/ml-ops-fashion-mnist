import io

import numpy as np
import pytest
from PIL import Image


def png_bytes(arr: np.ndarray, mode: str = "L") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode=mode).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mnist_like_png() -> bytes:
    """28x28 dark background with a light square (Fashion-MNIST style)."""
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[6:22, 8:20] = 200
    return png_bytes(arr)
