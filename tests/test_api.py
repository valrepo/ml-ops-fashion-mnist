import io

import pytest
import torch
from fastapi.testclient import TestClient

from fashion_mlops import CLASS_NAMES
from fashion_mlops.api import create_app
from fashion_mlops.inference import ModelService
from fashion_mlops.model import LeNet5
from fashion_mlops.storage import ModelNotFoundError, StorageError


def checkpoint_bytes() -> bytes:
    torch.manual_seed(0)
    model = LeNet5()
    buf = io.BytesIO()
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": {"num_classes": 10},
        "class_names": CLASS_NAMES,
        "run_id": "abc123",
        "val_accuracy": 0.9,
        "created_at": "2026-01-01T00:00:00+00:00",
    }, buf)
    return buf.getvalue()


class FakeStore:
    bucket = "fashion-minst-bucket"

    def __init__(self, objects: dict[str, bytes] | None = None, down: bool = False):
        self.objects = objects or {}
        self.down = down

    def resolve_key(self, selector):
        if self.down:
            raise StorageError("MinIO unreachable")
        if selector == "latest":
            if not self.objects:
                raise ModelNotFoundError("model/latest.json missing")
            return sorted(self.objects)[-1]
        return f"model/{selector}/model.pt"

    def download(self, key):
        if key not in self.objects:
            raise ModelNotFoundError(key)
        return self.objects[key]


def make_client(store: FakeStore) -> TestClient:
    service = ModelService(store_factory=lambda: store)
    return TestClient(create_app(service=service))


@pytest.fixture
def client():
    with make_client(FakeStore({"model/abc123/model.pt": checkpoint_bytes()})) as c:
        yield c


def test_health_with_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model"]["object_key"] == "model/abc123/model.pt"
    assert body["model"]["run_id"] == "abc123"


def test_predict_returns_class(client, mnist_like_png):
    resp = client.post("/predict", files={"file": ("x.png", mnist_like_png, "image/png")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["class_name"] in CLASS_NAMES
    assert body["class_name"] == CLASS_NAMES[body["class_id"]]
    assert set(body["probabilities"]) == set(CLASS_NAMES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_non_image(client):
    resp = client.post("/predict", files={"file": ("x.png", b"garbage", "image/png")})
    assert resp.status_code == 422
    assert "image" in resp.json()["detail"].lower()


def test_predict_rejects_wrong_content_type(client):
    resp = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422


def test_no_model_yet_gives_503(mnist_like_png):
    with make_client(FakeStore({})) as c:
        assert c.get("/health").json()["model_loaded"] is False
        resp = c.post("/predict", files={"file": ("x.png", mnist_like_png, "image/png")})
        assert resp.status_code == 503
        assert "Train a model first" in resp.json()["detail"]


def test_storage_down_gives_503(mnist_like_png):
    with make_client(FakeStore(down=True)) as c:
        body = c.get("/health").json()
        assert body["model_loaded"] is False and "unreachable" in body["error"]
        resp = c.post("/predict", files={"file": ("x.png", mnist_like_png, "image/png")})
        assert resp.status_code == 503


def test_model_loaded_on_demand_after_training(mnist_like_png):
    store = FakeStore({})
    with make_client(store) as c:
        assert c.get("/health").json()["model_loaded"] is False
        store.objects["model/run9/model.pt"] = checkpoint_bytes()  # training finishes
        resp = c.post("/predict", files={"file": ("x.png", mnist_like_png, "image/png")})
        assert resp.status_code == 200

