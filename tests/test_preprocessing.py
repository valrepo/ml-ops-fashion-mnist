import io

import numpy as np
import pytest
from PIL import Image

from conftest import png_bytes
from fashion_mlops.preprocessing import (
    MEAN, STD, InvalidImageError, preprocess_bytes, to_fashion_mnist_array,
)


def test_output_shape_and_dtype(mnist_like_png):
    x = preprocess_bytes(mnist_like_png)
    assert x.shape == (1, 1, 28, 28)
    assert x.dtype == np.float32


def test_matches_training_normalisation(mnist_like_png):
    """A 28x28 grayscale image must be processed exactly like torchvision ToTensor+Normalize."""
    import torch
    from fashion_mlops.data import TRANSFORM

    img = Image.open(io.BytesIO(mnist_like_png))
    expected = TRANSFORM(img).numpy()[None]
    np.testing.assert_allclose(preprocess_bytes(mnist_like_png), expected, atol=1e-6)
    assert torch.is_tensor(TRANSFORM(img))


def test_rgb_large_image_resized():
    arr = np.zeros((100, 60, 3), dtype=np.uint8)
    arr[20:80, 10:50] = (180, 40, 40)
    out = to_fashion_mnist_array(Image.fromarray(arr, "RGB"))
    assert out.shape == (28, 28)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_light_background_is_inverted():
    arr = np.full((28, 28), 255, dtype=np.uint8)
    arr[8:20, 8:20] = 0  # dark object on white
    out = to_fashion_mnist_array(Image.fromarray(arr, "L"))
    assert out[0, 0] == pytest.approx(0.0)    # background now dark
    assert out[14, 14] == pytest.approx(1.0)  # object now light
    raw = to_fashion_mnist_array(Image.fromarray(arr, "L"), auto_invert=False)
    assert raw[0, 0] == pytest.approx(1.0)


def test_transparent_png_handled():
    arr = np.zeros((40, 40, 4), dtype=np.uint8)
    arr[10:30, 10:30] = (0, 0, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, format="PNG")
    x = preprocess_bytes(buf.getvalue())
    assert x.shape == (1, 1, 28, 28)
    # transparent area -> white -> inverted -> dark background
    assert x[0, 0, 0, 0] == pytest.approx((0 - MEAN) / STD, abs=1e-5)


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"%PDF-1.4 fake"])
def test_invalid_bytes_rejected(data):
    with pytest.raises(InvalidImageError):
        preprocess_bytes(data)


def test_too_large_rejected(monkeypatch):
    import fashion_mlops.preprocessing as p
    monkeypatch.setattr(p, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(InvalidImageError, match="too large"):
        p.preprocess_bytes(png_bytes(np.zeros((28, 28))))
