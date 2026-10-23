# FastAPI prediction service (also used by the minio-init job).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app
# CPU-only PyTorch (same layer as the trainer image, so it is built once and cached)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "fashion_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
