"""MinIO model storage.

Layout inside the bucket
    model/<mlflow-run-id>/model.pt     one immutable object per training run
    model/latest.json                  pointer to the most recent successful run
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from .config import MODEL_PREFIX, StorageSettings

log = logging.getLogger(__name__)

MODEL_FILENAME = "model.pt"
LATEST_POINTER = f"{MODEL_PREFIX}latest.json"


class StorageError(RuntimeError):
    """MinIO is unreachable or the request failed."""


class ModelNotFoundError(StorageError):
    """The requested model object does not exist."""


def model_key(run_id: str) -> str:
    return f"{MODEL_PREFIX}{run_id}/{MODEL_FILENAME}"


@dataclass
class ModelStore:
    settings: StorageSettings

    def __post_init__(self) -> None:
        self.client = Minio(
            self.settings.endpoint,
            access_key=self.settings.access_key,
            secret_key=self.settings.secret_key,
            secure=self.settings.secure,
        )

    @property
    def bucket(self) -> str:
        return self.settings.bucket

    def _wrap(self, action: str, exc: Exception) -> StorageError:
        if isinstance(exc, S3Error) and exc.code in {"NoSuchKey", "NoSuchBucket"}:
            return ModelNotFoundError(f"{action}: {exc.code} ({exc.bucket_name}/{exc.object_name})")
        return StorageError(
            f"{action} failed - is MinIO reachable at {self.settings.endpoint}? ({exc})"
        )

    def ensure_bucket(self) -> None:
        """Create the bucket if needed (prefixes are implicit in S3/MinIO)."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                log.info("Created MinIO bucket %s", self.bucket)
        except (S3Error, Urllib3HTTPError, OSError) as exc:
            raise self._wrap("ensure_bucket", exc) from exc

    def upload_model(self, local_path: Path, run_id: str, metadata: dict) -> str:
        """Upload a model file under ``model/<run_id>/`` and update ``latest.json``."""
        key = model_key(run_id)
        try:
            self.ensure_bucket()
            self.client.fput_object(
                self.bucket, key, str(local_path), content_type="application/octet-stream"
            )
            # Read back to confirm the object really landed before anyone is notified.
            self.client.stat_object(self.bucket, key)
            pointer = json.dumps({"run_id": run_id, "object_key": key, **metadata}, indent=2)
            self.client.put_object(
                self.bucket,
                LATEST_POINTER,
                io.BytesIO(pointer.encode()),
                length=len(pointer.encode()),
                content_type="application/json",
            )
        except (S3Error, Urllib3HTTPError, OSError) as exc:
            raise self._wrap("upload_model", exc) from exc
        return key

    def resolve_key(self, selector: str = "latest") -> str:
        """Map a selector to an object key.
        ``latest`` -> key stored in ``model/latest.json``;
        a key ending with ``.pt`` -> used as-is;
        anything else -> treated as an MLflow run id.
        """
        if selector in ("", "latest"):
            try:
                resp = self.client.get_object(self.bucket, LATEST_POINTER)
                try:
                    return json.loads(resp.read())["object_key"]
                finally:
                    resp.close()
                    resp.release_conn()
            except (S3Error, Urllib3HTTPError, OSError) as exc:
                raise self._wrap("resolve latest model", exc) from exc
        if selector.endswith(".pt"):
            return selector
        return model_key(selector)

    def download(self, key: str) -> bytes:
        try:
            resp = self.client.get_object(self.bucket, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()
        except (S3Error, Urllib3HTTPError, OSError) as exc:
            raise self._wrap(f"download {key}", exc) from exc


def main() -> int:
    """``python -m fashion_mlops.storage``."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        store = ModelStore(StorageSettings.from_env())
        store.ensure_bucket()
    except Exception as exc:  # noqa: BLE001
        log.error("Storage init failed: %s", exc)
        return 1
    log.info("Bucket %s ready (models go under %s)", store.bucket, MODEL_PREFIX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
