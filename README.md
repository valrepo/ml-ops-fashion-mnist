# Fashion-MNIST MLOps: Train, Track, Store and Serve

An end-to-end mini MLOps project. It trains a **LeNet-5** CNN on **Fashion-MNIST**, tracks every run in **MLflow**, stores a versioned model in **MinIO**, sends a **webhook** once the model is stored, and serves single-image predictions through **FastAPI** with a **Streamlit** UI. **Docker Compose** runs every service.

```text
configs/train.toml (+ env / --set overrides)
        │
        ▼
 trainer ──► MLflow (params, per-epoch metrics, config, reports, model.pt)
        │
        ├──► MinIO  fashion-minst-bucket/model/<run-id>/model.pt  (+ model/latest.json)
        │
        └──► webhook POST {event: training.completed, run_id, bucket, object_key, val_accuracy…}
                     (sent only after the MinIO upload is confirmed)

 Streamlit UI ──HTTP──► FastAPI /predict ──(loads selected model from MinIO)──► class + probabilities
```

| Service            | URL (host)                         | Purpose                                    |
|--------------------|------------------------------------|--------------------------------------------|
| Streamlit UI       | http://localhost:8501              | upload an image, see the prediction        |
| FastAPI            | http://localhost:8000/docs         | inference API (OpenAPI docs)               |
| MLflow             | http://localhost:5050              | experiment tracking UI                     |
| MinIO console      | http://localhost:9001              | browse the bucket (log in with `.env` creds)|
| MinIO S3 API       | http://localhost:9000              | object storage                             |
| Webhook receiver   | http://localhost:8088/webhook      | local test receiver (GET lists payloads)   |

> **Note on the bucket name:** the spelling `fashion-minst-bucket` ("minst", not "mnist") is intentional. It is the exact name the assignment requires.

---

## Project layout

```text
pyproject.toml              project metadata + dependencies (extras: train, api, ui, dev)
configs/train.toml          example training configuration
src/fashion_mlops/
  config.py                 TOML/env/CLI config loading + validation, MinIO settings
  model.py                  LeNet-5
  data.py                   Fashion-MNIST download, seeded train/val split, loaders
  preprocessing.py          image preprocessing shared by training and inference
  train.py                  training pipeline (MLflow → MinIO → webhook)  [fashion-train]
  storage.py                MinIO model store (upload, latest pointer, download)
  webhook.py                webhook sender
  inference.py              model loading from MinIO + prediction
  api.py                    FastAPI app
ui/streamlit_app.py         Streamlit front-end (calls the API only)
scripts/webhook_receiver.py tiny stdlib webhook receiver for testing
docker/trainer.Dockerfile   training job image
docker/api.Dockerfile       FastAPI image (also runs the minio-init job)
docker/ui.Dockerfile        Streamlit image (no torch)
docker-compose.yml          minio, minio-init, mlflow, webhook-receiver, trainer, api, ui
tests/                      pytest suite (config, preprocessing, API, model, webhook)
```

---

## 1. Setup

Prerequisites: Docker with Compose v2. For local development you also need Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
cp .env.example .env
# Edit .env and set MINIO_ROOT_PASSWORD to a long random value, for example:
#   sed -i '' "s/change-me-to-a-long-random-string/$(openssl rand -hex 16)/" .env   # macOS
```

`.env` is git-ignored, so no credentials are committed. Compose refuses to start if the MinIO credentials are missing.

Local development environment (optional, needed for running the tests or training outside Docker):

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run pytest
```

---

## 2. Startup sequence (Docker Compose)

```bash
# 1) Infrastructure + prediction stack.
#    minio-init creates the bucket; the API starts even without a model.
docker compose up -d --build

# 2) Train once. This is a one-shot job: it waits for MinIO/MLflow, trains, logs to MLflow,
#    uploads to MinIO, then sends the webhook. Exit code 0 = success.
docker compose --profile train run --rm trainer

# 3) Open the UI at http://localhost:8501 and upload an image.
```

The API loads `model/latest.json` at startup. If no model exists yet (step 2 not run), the API stays up and reports `model_loaded: false` on `/health`. It retries loading on the next `/predict` call, so you don't need to restart anything after training. Until then the Streamlit sidebar shows a *"no model is loaded yet – run training first"* message. Once a model is loaded, the API keeps it. After training a newer model, run `docker compose restart api` to serve it.

A stricter sequence, if you prefer to start serving only once a model exists:

```bash
docker compose up -d minio minio-init mlflow webhook-receiver
docker compose --profile train run --rm trainer
docker compose up -d api ui
```

Stop everything with `docker compose down`. Add `-v` to also delete the MinIO/MLflow/dataset volumes.

---

## 3. Configuring training

Settings are resolved in this order, with later sources winning:

1. built-in defaults (`src/fashion_mlops/config.py`)
2. TOML file: `--config path.toml` (default `configs/train.toml`, or `$TRAIN_CONFIG`)
3. environment variables `TRAIN__<SECTION>__<KEY>`, e.g. `TRAIN__TRAINING__EPOCHS=5`
4. command line `--set section.key=value` (repeatable)

Unknown keys and invalid values (e.g. `optimizer = "lbfgs"`, `val_split = 1.5`) are rejected before training starts.

```bash
# Docker (configs/ is mounted read-only, so edit the TOML without rebuilding)
docker compose --profile train run --rm trainer --config configs/train.toml \
    --set training.epochs=15 --set training.optimizer=sgd --set training.learning_rate=0.05

docker compose --profile train run --rm -e TRAIN__TRAINING__SEED=7 trainer

# Locally, against the Compose infrastructure
set -a; source .env; set +a
export MLFLOW_TRACKING_URI=http://localhost:5050 MINIO_ENDPOINT=localhost:9000 \
       MINIO_ACCESS_KEY=$MINIO_ROOT_USER MINIO_SECRET_KEY=$MINIO_ROOT_PASSWORD \
       WEBHOOK_URL=http://localhost:8088/webhook
uv run fashion-train --set training.epochs=3
```

### Training parameters

| Key | Default | Description |
|---|---|---|
| `experiment.name` | `fashion-mnist-lenet5` | MLflow experiment |
| `experiment.run_name` | `""` | MLflow run name (empty = auto) |
| `experiment.tags` | `{}` | extra MLflow tags (CLI/env: `k=v,k2=v2`) |
| `data.data_dir` | `data` | dataset download/cache directory |
| `data.val_split` | `0.1` | fraction of the 60k training set used for validation (seeded split) |
| `data.num_workers` | `2` | DataLoader workers |
| `data.pin_memory` | `false` | DataLoader pin_memory |
| `training.epochs` | `10` | maximum epochs |
| `training.batch_size` | `128` | |
| `training.learning_rate` | `0.001` | |
| `training.optimizer` | `adam` | `sgd`, `adam`, `adamw`, `rmsprop` |
| `training.momentum` | `0.9` | for `sgd` / `rmsprop` |
| `training.weight_decay` | `0.0` | |
| `training.loss` | `cross_entropy` | `cross_entropy` or `nll` (log-softmax + NLL) |
| `training.scheduler` | `none` | `none`, `step`, `cosine`, `plateau` |
| `training.step_size` / `training.gamma` | `5` / `0.5` | step-scheduler period / decay (gamma also used by plateau) |
| `training.early_stopping_patience` | `3` | epochs without val-loss improvement before stopping (0 = off). The best checkpoint is kept. |
| `training.seed` | `42` | seeds Python/NumPy/PyTorch, the train/val split and the shuffling |
| `training.device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |

**Reproducibility:** each run logs the fully resolved configuration (`config/resolved_config.json`) and the TOML file that was used. Every value is also logged as an MLflow parameter, together with the torch/python versions and the device. Re-running with the same config and seed reproduces the run; on GPU, small differences from non-deterministic kernels are possible.

### Infrastructure environment variables

| Variable | Used by | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | trainer | `http://mlflow:5000` in Compose |
| `MINIO_ENDPOINT` | trainer, api | `minio:9000` in Compose |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | trainer, api | from `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` in `.env` |
| `MINIO_BUCKET` | trainer, api | `fashion-minst-bucket` |
| `MINIO_SECURE` | trainer, api | `false` |
| `MODEL_SELECTOR` | api | `latest` |
| `WEBHOOK_URL` | trainer | `http://webhook-receiver:8088/webhook` (unset = no webhook) |
| `API_URL` | ui | `http://api:8000` |

---

## 4. MLflow

Open **http://localhost:5050**, then the experiment **`fashion-mnist-lenet5`** in the left panel. Each training run is one row.

- **Parameters:** every config value (`training.epochs`, `training.batch_size`, `training.learning_rate`, `training.optimizer`, `training.loss`, `training.seed`, `data.val_split`, …) plus the dataset split sizes.
- **Metrics (per epoch):** `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy_epoch`, `learning_rate`. Click a metric to see its curve. Select several runs and press **Compare** to overlay them.
- **Final metrics** (best checkpoint): `val_accuracy`, `val_precision_macro`, `val_recall_macro`, `val_f1_macro`, `test_accuracy`, `test_loss`, `best_epoch`, `train_seconds`.
- **Artifacts:**
  - `model/model.pt`: the trained checkpoint
  - `model/minio_reference.json`: bucket, object key and `s3://` URI in MinIO
  - `config/`: the resolved config and the TOML file
  - `evaluation/val_confusion_matrix.json` and `evaluation/val_classification_report.txt`
- **Tags:** `minio_bucket`, `minio_object_key`, `device`, `torch_version`, `num_parameters`, `early_stopped`.

The MLflow server stores its data in the `mlflow-data` volume (SQLite backend + proxied artifact store), so runs survive restarts.

---

## 5. MinIO model storage

- Bucket: **`fashion-minst-bucket`**. It is created by the `minio-init` job (`python -m fashion_mlops.storage`) and also by the trainer if missing.
- Object per run: **`model/<mlflow-run-id>/model.pt`**. Run IDs are unique, so a new training run never overwrites an older model.
- Pointer: **`model/latest.json`** holds `{run_id, object_key, val_accuracy, test_accuracy, created_at}` of the latest successful run.
- The checkpoint (`torch.save` dict) contains the `state_dict`, the model config, the class names, the preprocessing constants, the run id and the validation accuracy. The API loads it with `weights_only=True`, so no arbitrary pickles are executed.

**How the API chooses a model:** the `MODEL_SELECTOR` environment variable, read when the API loads its model, can be:

| value | loads |
|---|---|
| `latest` (default) | the object named in `model/latest.json` |
| an MLflow run id, e.g. `3f2c…` | `model/<run-id>/model.pt` |
| a full key ending in `.pt` | that object |

```bash
# pin the API to a given run
MODEL_SELECTOR=<run-id> docker compose up -d api
```

Error handling: if MinIO is unreachable or the object doesn't exist, training fails with a clear `StorageError`. The trainer checks the bucket *before* training, so no time is wasted. The API answers `503` with an explanation, and `/health` shows the error message.

Browse the objects in the MinIO console at http://localhost:9001 with the credentials from `.env`.

---

## 6. Training-complete webhook

After training, MLflow logging **and** a confirmed MinIO upload (a `stat_object` read-back), the trainer POSTs this JSON to `WEBHOOK_URL`:

```json
{
  "event": "training.completed",
  "status": "success",
  "run_id": "9b1d7c3e5f2a4e0c8d6b7a1f2e3d4c5b",
  "experiment_name": "fashion-mnist-lenet5",
  "experiment_id": "1",
  "mlflow_run_url": "http://mlflow:5000/#/experiments/1/runs/9b1d7c3e5f2a4e0c8d6b7a1f2e3d4c5b",
  "model": {
    "bucket": "fashion-minst-bucket",
    "object_key": "model/9b1d7c3e5f2a4e0c8d6b7a1f2e3d4c5b/model.pt"
  },
  "metrics": { "val_accuracy": 0.9012, "test_accuracy": 0.8968, "best_epoch": 10 },
  "params": { "epochs": 10, "batch_size": 128, "learning_rate": 0.001,
              "optimizer": "adam", "loss": "cross_entropy", "seed": 42 },
  "completed_at": "2026-10-11T19:42:03.512345+00:00"
}
```

- If any step fails (config, data, training, MLflow, MinIO), **no webhook is sent**. The error is logged with a traceback, the MLflow run is marked `FAILED`, and the process exits with code `1`.
- If the webhook itself can't be delivered, the model remains stored but the trainer logs the error and exits with code `3`.
- An empty `WEBHOOK_URL` disables notifications.

**Testing notifications:** the Compose stack includes `webhook-receiver`:

```bash
docker compose logs -f webhook-receiver          # payloads are printed as they arrive
curl http://localhost:8088/webhook               # list received payloads
cat webhook_payloads/payloads.jsonl              # persisted on the host
```

Without Docker: `python scripts/webhook_receiver.py`, then `WEBHOOK_URL=http://localhost:8088/webhook`. You can also point `WEBHOOK_URL` at a public inspector such as webhook.site.

---

## 7. Prediction API (FastAPI)

Interactive docs: **http://localhost:8000/docs**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status":"ok","model_loaded":bool,"model":{…},"selector":…,"error":…}` |
| POST | `/predict?auto_invert=true` | multipart form, field **`file`** = one image |

```bash
curl -F "file=@sample.png" http://localhost:8000/predict
```

```json
{
  "class_id": 9,
  "class_name": "Ankle boot",
  "confidence": 0.987,
  "probabilities": {"T-shirt/top": 0.0001, "Trouser": 0.0, "...": "...", "Ankle boot": 0.987},
  "model": {"run_id": "9b1d…", "bucket": "fashion-minst-bucket", "object_key": "model/9b1d…/model.pt", "val_accuracy": 0.90}
}
```

**Preprocessing** (`src/fashion_mlops/preprocessing.py`) matches training: the image is converted to grayscale (transparency flattened onto white) and padded to a square, then resized to 28×28, scaled to [0, 1] and normalised with the training mean/std (0.2860 / 0.3530). Fashion-MNIST items are light on a dark background, so images with a mostly light border (e.g. product photos on white) are inverted. Pass `auto_invert=false` to disable this. A test checks that a 28×28 image gives exactly the same tensor as the training transform.

**Errors:**

| code | when |
|---|---|
| `415` | non-image content type |
| `422` | missing file, empty or corrupt file, unsupported format, or file larger than 5 MB |
| `503` | no model available or MinIO unreachable |

---

## 8. Streamlit UI

Open **http://localhost:8501** and:

1. check the sidebar for the backend and model status (run id, MinIO object, validation accuracy);
2. upload one image (PNG/JPEG/BMP/GIF/TIFF/WEBP) to see a preview;
3. press **Predict** to see the predicted label, the confidence and the probability of each class.

The UI only calls the FastAPI service, with no model code of its own. Unreachable API, missing model and invalid image each get their own message.

Need a test image? Export one from the dataset:

```bash
uv run python -c "
from torchvision.datasets import FashionMNIST
ds = FashionMNIST('data', train=False, download=True)
for i in range(5): img, y = ds[i]; img.save(f'sample_{i}_{ds.classes[y].replace(\"/\", \"-\")}.png')
"
```

---

## 9. Tests

```bash
uv run pytest
```

The tests cover:

- config precedence, type coercion and validation;
- preprocessing (shape, parity with the training transform, inversion, transparency, invalid/oversized files);
- the API with a fake MinIO store (health, prediction response, 415/422/503 handling, on-demand loading);
- the model's output shape and every optimizer × loss × scheduler combination;
- the webhook (skipped without URL, JSON POST, delivery failure).

---

## Notes

- **MinIO image:** MinIO stopped publishing official `minio/minio` Docker images in October 2025, and the upstream repository is now archived. Compose therefore uses `pgsty/minio`, a maintained community build of the same server with the same CLI and S3 API. Any S3-compatible MinIO build works if you change the `image:` line.
- **Port 5050 for MLflow:** on macOS, port 5000 is used by AirPlay Receiver. Inside the Compose network MLflow still listens on `mlflow:5000`.
- **Port conflicts:** every host port can be changed in `.env` (`API_PORT`, `UI_PORT`, `MLFLOW_PORT`, `MINIO_API_PORT`, `MINIO_CONSOLE_PORT`, `WEBHOOK_PORT`). Services talk to each other on the internal Compose network, so only the host-side URLs change.
- **CPU-only PyTorch** in the images keeps them around 1 GB. A full 10-epoch run takes a few minutes on a laptop CPU, and LeNet-5 reaches about 89–91 % validation accuracy.
