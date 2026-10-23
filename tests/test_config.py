from pathlib import Path

import pytest

from fashion_mlops.config import Config, ConfigError, StorageSettings, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "train.toml"


def test_defaults_are_valid():
    cfg = load_config(env={})
    assert cfg.training.epochs == 10
    assert cfg.training.optimizer == "adam"


def test_example_toml_loads():
    cfg = load_config(EXAMPLE, env={})
    assert cfg.experiment.name == "fashion-mnist-lenet5"
    assert cfg.experiment.tags["model"] == "lenet5"
    assert 0 < cfg.data.val_split < 1


def test_precedence_toml_env_cli(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[training]\nepochs = 3\nbatch_size = 32\nlearning_rate = 0.01\n")
    cfg = load_config(
        f,
        overrides=["training.batch_size=256"],
        env={"TRAIN__TRAINING__EPOCHS": "7", "TRAIN__TRAINING__BATCH_SIZE": "64", "UNRELATED": "x"},
    )
    assert cfg.training.epochs == 7            # env beats TOML
    assert cfg.training.batch_size == 256      # CLI beats env
    assert cfg.training.learning_rate == 0.01  # TOML beats default


def test_type_coercion():
    cfg = load_config(env={}, overrides=[
        "data.pin_memory=true", "training.learning_rate=5e-4", "experiment.tags=a=1,b=2",
    ])
    assert cfg.data.pin_memory is True
    assert cfg.training.learning_rate == pytest.approx(5e-4)
    assert cfg.experiment.tags == {"a": "1", "b": "2"}


@pytest.mark.parametrize("override", [
    "training.optimizer=lbfgs",
    "training.loss=mse",
    "training.epochs=0",
    "training.learning_rate=-1",
    "data.val_split=1.5",
    "training.epochs=abc",
    "training.unknown=1",
    "nosection.key=1",
    "noequals",
])
def test_invalid_values_rejected(override):
    with pytest.raises(ConfigError):
        load_config(env={}, overrides=[override])


def test_unknown_toml_key_rejected(tmp_path):
    f = tmp_path / "bad.toml"
    f.write_text("[training]\nepochz = 3\n")
    with pytest.raises(ConfigError, match="epochz"):
        load_config(f, env={})


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.toml", env={})


def test_flat_params_excludes_tags():
    params = Config().flat_params()
    assert params["training.seed"] == 42
    assert "experiment.tags" not in params


def test_storage_settings_require_credentials(monkeypatch):
    for var in ("MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match="credentials"):
        StorageSettings.from_env()
    monkeypatch.setenv("MINIO_ACCESS_KEY", "u")
    monkeypatch.setenv("MINIO_SECRET_KEY", "p")
    monkeypatch.delenv("MINIO_BUCKET", raising=False)
    assert StorageSettings.from_env().bucket == "fashion-minst-bucket"
