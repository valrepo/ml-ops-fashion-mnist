# Training job: train -> MLflow -> MinIO -> webhook.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app
# CPU-only PyTorch (same layer as the api image, so it is built once and cached)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[train]"
COPY configs ./configs
RUN mkdir -p /app/data && chown app /app/data
USER app
ENTRYPOINT ["fashion-train"]
CMD ["--config", "configs/train.toml"]
