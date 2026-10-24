import pytest
import torch

from fashion_mlops.config import load_config
from fashion_mlops.model import LeNet5
from fashion_mlops.storage import model_key
from fashion_mlops.train import build_loss, build_optimizer, build_scheduler
from fashion_mlops.webhook import WebhookError, send_webhook


def test_lenet_output_shape():
    out = LeNet5()(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


@pytest.mark.parametrize("opt", ["sgd", "adam", "adamw", "rmsprop"])
@pytest.mark.parametrize("loss", ["cross_entropy", "nll"])
@pytest.mark.parametrize("sched", ["none", "step", "cosine", "plateau"])
def test_builders(opt, loss, sched):
    cfg = load_config(env={}, overrides=[
        f"training.optimizer={opt}", f"training.loss={loss}", f"training.scheduler={sched}",
    ])
    model = LeNet5()
    optimizer = build_optimizer(cfg, model.parameters())
    criterion = build_loss(cfg)
    build_scheduler(cfg, optimizer)
    value = criterion(model(torch.randn(2, 1, 28, 28)), torch.tensor([0, 3]))
    assert value.ndim == 0 and torch.isfinite(value)


def test_model_key_is_versioned():
    assert model_key("abc") == "model/abc/model.pt"
    assert model_key("abc") != model_key("def")


def test_webhook_skipped_without_url(monkeypatch):
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    assert send_webhook({"event": "x"}) is False


def test_webhook_posts_json(monkeypatch):
    sent = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json)
        return Resp()

    monkeypatch.setattr("fashion_mlops.webhook.requests.post", fake_post)
    assert send_webhook({"event": "training.completed"}, url="http://hook")
    assert sent == {"url": "http://hook", "json": {"event": "training.completed"}}


def test_webhook_failure_raises(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("fashion_mlops.webhook.requests.post", boom)
    with pytest.raises(WebhookError):
        send_webhook({"event": "x"}, url="http://hook")
