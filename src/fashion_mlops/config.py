"""Training configuration: defaults < TOML file < env vars < CLI ``--set`` overrides.

Infrastructure settings (MLflow, MinIO, webhook) are read from environment
variables only, so that endpoints and credentials never live in the repository.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

ENV_PREFIX = "TRAIN__"

OPTIMIZERS = {"sgd", "adam", "adamw", "rmsprop"}
LOSSES = {"cross_entropy", "nll"}
SCHEDULERS = {"none", "step", "cosine", "plateau"}
DEVICES = {"auto", "cpu", "cuda", "mps"}


class ConfigError(ValueError):
    """Raised for invalid or unknown configuration values."""


@dataclass
class ExperimentConfig:
    name: str = "fashion-mnist-lenet5"
    run_name: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class DataConfig:
    data_dir: str = "data"
    val_split: float = 0.1
    num_workers: int = 2
    pin_memory: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    loss: str = "cross_entropy"
    scheduler: str = "none"
    step_size: int = 5
    gamma: float = 0.5
    early_stopping_patience: int = 3
    seed: int = 42
    device: str = "auto"


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_params(self) -> dict[str, Any]:
        """used for MLflow params """
        out: dict[str, Any] = {}
        for section, values in self.to_dict().items():
            for key, value in values.items():
                if isinstance(value, dict):
                    continue  # tags are logged as MLflow tags, not params
                out[f"{section}.{key}"] = value
        return out

    def validate(self) -> None:
        t, d = self.training, self.data
        _check_choice("training.optimizer", t.optimizer, OPTIMIZERS)
        _check_choice("training.loss", t.loss, LOSSES)
        _check_choice("training.scheduler", t.scheduler, SCHEDULERS)
        _check_choice("training.device", t.device, DEVICES)
        if t.epochs < 1:
            raise ConfigError("training.epochs must be >= 1")
        if t.batch_size < 1:
            raise ConfigError("training.batch_size must be >= 1")
        if t.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be > 0")
        if t.early_stopping_patience < 0:
            raise ConfigError("training.early_stopping_patience must be >= 0")
        if not 0.0 < d.val_split < 1.0:
            raise ConfigError("data.val_split must be in (0, 1)")
        if d.num_workers < 0:
            raise ConfigError("data.num_workers must be >= 0")


def _check_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        raise ConfigError(f"{name}={value!r} is not one of {sorted(choices)}")


def _coerce(raw: Any, target_type: Any, name: str) -> Any:
    if target_type is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if isinstance(raw, str) and raw.strip().lower() in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"{name}: cannot interpret {raw!r} as a boolean")
    if target_type in (int, float, str):
        try:
            if target_type is int and isinstance(raw, float) and not raw.is_integer():
                raise ValueError
            return target_type(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name}: cannot interpret {raw!r} as {target_type.__name__}") from exc
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):  # dict from env/CLI: "k1=v1,k2=v2"
        pairs = [p.split("=", 1) for p in raw.split(",") if p.strip()]
        if any(len(p) != 2 for p in pairs):
            raise ConfigError(f"{name}: expected 'key=value,key2=value2', got {raw!r}")
        return {k.strip(): v.strip() for k, v in pairs}
    raise ConfigError(f"{name}: unsupported value {raw!r}")


def _set(cfg: Config, dotted: str, raw: Any) -> None:
    parts = dotted.strip().split(".")
    if len(parts) != 2:
        raise ConfigError(f"Override key must look like 'section.key', got {dotted!r}")
    section_name, key = parts
    section = getattr(cfg, section_name, None)
    if section is None or not is_dataclass(section):
        raise ConfigError(f"Unknown config section {section_name!r}")
    hints = get_type_hints(type(section))
    if key not in hints:
        raise ConfigError(f"Unknown config key {dotted!r}")
    setattr(section, key, _coerce(raw, hints[key], dotted))


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Build a validated :class:`Config`.

    ``overrides`` are ``section.key=value`` strings (from ``--set``).
    ``env`` defaults to ``os.environ``; variables named ``TRAIN__SECTION__KEY`` apply.
    """
    cfg = Config()

    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        for section_name, values in data.items():
            if not isinstance(values, dict):
                raise ConfigError(f"Top-level key {section_name!r} must be a [section]")
            for key, value in values.items():
                _set(cfg, f"{section_name}.{key}", value)

    env = os.environ if env is None else env
    for name, value in sorted(env.items()):
        if name.startswith(ENV_PREFIX):
            dotted = name[len(ENV_PREFIX):].lower().replace("__", ".")
            _set(cfg, dotted, value)

    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"--set expects section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set(cfg, key, value)

    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# Infrastructure settings (environment only)
# --------------------------------------------------------------------------- #

DEFAULT_BUCKET = "fashion-minst-bucket"
MODEL_PREFIX = "model/"


@dataclass
class StorageSettings:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    bucket: str

    @classmethod
    def from_env(cls) -> "StorageSettings":
        access_key = os.environ.get("MINIO_ACCESS_KEY") or os.environ.get("MINIO_ROOT_USER")
        secret_key = os.environ.get("MINIO_SECRET_KEY") or os.environ.get("MINIO_ROOT_PASSWORD")
        if not access_key or not secret_key:
            raise ConfigError(
                "MinIO credentials missing: set MINIO_ACCESS_KEY and MINIO_SECRET_KEY "
                "(see .env.example)"
            )
        return cls(
            endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=access_key,
            secret_key=secret_key,
            secure=os.environ.get("MINIO_SECURE", "false").lower() in {"1", "true", "yes"},
            bucket=os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET),
        )
