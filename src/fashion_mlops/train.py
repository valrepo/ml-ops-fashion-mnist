"""Train LeNet-5 on Fashion-MNIST, track in MLflow, store in MinIO, notify via webhook.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn

from . import CLASS_NAMES
from .config import Config, ConfigError, StorageSettings, load_config
from .data import build_dataloaders
from .model import LeNet5
from .preprocessing import IMAGE_SIZE, MEAN, STD
from .storage import ModelStore
from .webhook import WebhookError, send_webhook

log = logging.getLogger("fashion_mlops.train")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_optimizer(cfg: Config, params) -> torch.optim.Optimizer:
    t = cfg.training
    if t.optimizer == "sgd":
        return torch.optim.SGD(params, lr=t.learning_rate, momentum=t.momentum, weight_decay=t.weight_decay)
    if t.optimizer == "adam":
        return torch.optim.Adam(params, lr=t.learning_rate, weight_decay=t.weight_decay)
    if t.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=t.learning_rate, weight_decay=t.weight_decay)
    if t.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=t.learning_rate, momentum=t.momentum, weight_decay=t.weight_decay)
    raise ConfigError(f"Unknown optimizer {t.optimizer!r}")


def build_loss(cfg: Config) -> nn.Module:
    """Return a criterion that takes raw logits."""
    t = cfg.training
    if t.loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if t.loss == "nll":
        nll = nn.NLLLoss()
        return _LogSoftmaxThen(nll)
    raise ConfigError(f"Unknown loss {t.loss!r}")


class _LogSoftmaxThen(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.inner(torch.log_softmax(logits, dim=1), target)


def build_scheduler(cfg: Config, optimizer):
    t = cfg.training
    if t.scheduler == "none":
        return None
    if t.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=t.step_size, gamma=t.gamma)
    if t.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.epochs)
    if t.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=t.gamma, patience=1)
    raise ConfigError(f"Unknown scheduler {t.scheduler!r}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, seen = 0.0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        seen += x.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(targets)
    return total_loss / seen, float((y_pred == y_true).mean()), y_true, y_pred


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def run(cfg: Config, config_path: str | None, state: dict | None = None) -> dict:
    """Execute the full pipeline. Returns the webhook payload on success.
    """
    state = {} if state is None else state
    storage = ModelStore(StorageSettings.from_env())
    storage.ensure_bucket()  # fail fast if MinIO is unreachable, before spending time training

    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    log.info("Device: %s", device)

    loaders = build_dataloaders(cfg)
    log.info("Dataset sizes: %s", loaders.sizes)

    model = LeNet5().to(device)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)

    mlflow.set_experiment(cfg.experiment.name)
    with mlflow.start_run(run_name=cfg.experiment.run_name or None) as active:
        run_id = active.info.run_id
        state["run_id"] = run_id
        log.info("MLflow run %s (experiment %r)", run_id, cfg.experiment.name)

        mlflow.log_params(cfg.flat_params())
        mlflow.log_params({f"data.size_{k}": v for k, v in loaders.sizes.items()})
        mlflow.set_tags({
            **cfg.experiment.tags,
            "device": str(device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "num_parameters": sum(p.numel() for p in model.parameters()),
        })
        # Exact configuration used, for reproducibility.
        mlflow.log_dict(cfg.to_dict(), "config/resolved_config.json")
        if config_path:
            mlflow.log_artifact(config_path, artifact_path="config")

        best_val_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
        patience = cfg.training.early_stopping_patience
        start = time.time()
        for epoch in range(1, cfg.training.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, loaders.train, criterion, optimizer, device)
            va_loss, va_acc, _, _ = evaluate(model, loaders.val, criterion, device)
            lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(
                {"train_loss": tr_loss, "train_accuracy": tr_acc,
                 "val_loss": va_loss, "val_accuracy_epoch": va_acc, "learning_rate": lr},
                step=epoch,
            )
            log.info("epoch %d/%d  train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f  lr=%.2e",
                     epoch, cfg.training.epochs, tr_loss, tr_acc, va_loss, va_acc, lr)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(va_loss)
                else:
                    scheduler.step()

            if va_loss < best_val_loss:
                best_val_loss, best_epoch, bad_epochs = va_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                    mlflow.set_tag("early_stopped", "true")
                    break

        model.load_state_dict(best_state)
        train_seconds = time.time() - start

        # Final evaluation with the best checkpoint.
        val_loss, val_acc, y_true, y_pred = evaluate(model, loaders.val, criterion, device)
        test_loss, test_acc, _, _ = evaluate(model, loaders.test, criterion, device)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        final_metrics = {
            "val_accuracy": val_acc, "val_loss_best": val_loss,
            "val_precision_macro": float(precision), "val_recall_macro": float(recall),
            "val_f1_macro": float(f1), "test_accuracy": test_acc, "test_loss": test_loss,
            "best_epoch": best_epoch, "train_seconds": train_seconds,
        }
        mlflow.log_metrics(final_metrics)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        mlflow.log_dict({"labels": CLASS_NAMES, "matrix": cm.tolist()}, "evaluation/val_confusion_matrix.json")
        mlflow.log_text(
            classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0),
            "evaluation/val_classification_report.txt",
        )
        log.info("Final: val_acc=%.4f test_acc=%.4f (best epoch %d)", val_acc, test_acc, best_epoch)

        # Serialise the model with everything inference needs.
        created_at = datetime.now(timezone.utc).isoformat()
        checkpoint = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": {"num_classes": len(CLASS_NAMES)},
            "class_names": CLASS_NAMES,
            "preprocessing": {"image_size": IMAGE_SIZE, "mean": MEAN, "std": STD},
            "run_id": run_id,
            "val_accuracy": val_acc,
            "created_at": created_at,
        }
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pt"
            torch.save(checkpoint, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            object_key = storage.upload_model(
                model_path, run_id,
                metadata={"val_accuracy": val_acc, "test_accuracy": test_acc, "created_at": created_at},
            )
        model_ref = {
            "bucket": storage.bucket, "object_key": object_key,
            "uri": f"s3://{storage.bucket}/{object_key}", "endpoint": storage.settings.endpoint,
        }
        mlflow.log_dict(model_ref, "model/minio_reference.json")
        mlflow.set_tags({"minio_bucket": storage.bucket, "minio_object_key": object_key})
        log.info("Model uploaded to %s", model_ref["uri"])

        experiment_id = active.info.experiment_id

    # Only reached when training, MLflow logging and the MinIO upload all succeeded.
    tracking_uri = mlflow.get_tracking_uri()
    return {
        "event": "training.completed",
        "status": "success",
        "run_id": run_id,
        "experiment_name": cfg.experiment.name,
        "experiment_id": experiment_id,
        "mlflow_run_url": f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
        if tracking_uri.startswith("http") else None,
        "model": {"bucket": storage.bucket, "object_key": object_key},
        "metrics": {"val_accuracy": round(val_acc, 4), "test_accuracy": round(test_acc, 4),
                    "best_epoch": best_epoch},
        "params": {"epochs": cfg.training.epochs, "batch_size": cfg.training.batch_size,
                   "learning_rate": cfg.training.learning_rate, "optimizer": cfg.training.optimizer,
                   "loss": cfg.training.loss, "seed": cfg.training.seed},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", "-c", default=os.environ.get("TRAIN_CONFIG", "configs/train.toml"),
                   help="Path to a TOML config file (default: $TRAIN_CONFIG or configs/train.toml)")
    p.add_argument("--set", "-s", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="Override a config value; may be repeated")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=args.set)
    except ConfigError as exc:
        log.error("Invalid configuration: %s", exc)
        return 2

    state: dict = {}
    try:
        payload = run(cfg, args.config, state)
    except Exception as exc:  # report any failure, never send a success webhook
        log.exception("Training pipeline FAILED: %s", exc)
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        if state.get("run_id"):
            log.error("MLflow run %s marked as FAILED; no webhook sent.", state["run_id"])
        return 1

    log.info("Training-complete payload:\n%s", json.dumps(payload, indent=2))
    try:
        send_webhook(payload)
    except WebhookError as exc:
        log.error("Model is stored, but the success webhook could not be delivered: %s", exc)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
