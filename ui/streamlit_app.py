"""Streamlit front-end: upload an image, call the FastAPI backend, show the prediction."""

from __future__ import annotations

import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("API_TIMEOUT", "30"))

st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👕", layout="centered")


@st.cache_data(ttl=5, show_spinner=False)
def api_health() -> tuple[dict | None, str | None]:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def error_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


st.title("👕 Fashion-MNIST Classifier")
st.caption("LeNet-5 trained on Fashion-MNIST · served by FastAPI · model stored in MinIO")

with st.sidebar:
    st.header("Backend status")
    health, err = api_health()
    if err:
        st.error(f"API unreachable at `{API_URL}`.\n\n{err}")
    elif not health.get("model_loaded"):
        st.warning(
            "API is up but **no model is loaded yet**.\n\n"
            "Run training first:\n\n`docker compose --profile train run --rm trainer`\n\n"
            "The API will pick the model up on the next prediction."
        )
        if health.get("error"):
            st.caption(health["error"])
    else:
        info = health.get("model") or {}
        st.success("Model loaded")
        st.markdown(
            f"- **Run:** `{info.get('run_id', '?')}`\n"
            f"- **Object:** `{info.get('bucket')}/{info.get('object_key')}`\n"
            f"- **Val. accuracy:** {info.get('val_accuracy', 0):.2%}"
        )
    if st.button("Refresh status"):
        api_health.clear()
        st.rerun()
    st.divider()
    auto_invert = st.checkbox(
        "Auto-invert light backgrounds", value=True,
        help="Fashion-MNIST items are light on a dark background. Photos on a white "
             "background are inverted automatically when this is on.",
    )

st.markdown(
    "Upload a single image of a clothing item. Best results with Fashion-MNIST-style "
    "images (28×28 grayscale, item centred on a dark background); other images are "
    "converted to grayscale and resized."
)

uploaded = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp"]
)

if uploaded is not None:
    data = uploaded.getvalue()
    col_img, col_res = st.columns([1, 1.4], gap="large")
    with col_img:
        try:
            preview = Image.open(io.BytesIO(data))
            st.image(preview.resize((224, 224), Image.Resampling.NEAREST) if max(preview.size) <= 64 else preview,
                     caption=f"{uploaded.name} · {preview.size[0]}×{preview.size[1]}", width="stretch")
        except Exception:
            st.error("This file could not be opened as an image.")
            st.stop()

    with col_res:
        if st.button("🔮 Predict", type="primary", width="stretch"):
            with st.spinner("Asking the model..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        params={"auto_invert": str(auto_invert).lower()},
                        files={"file": (uploaded.name, data, uploaded.type or "application/octet-stream")},
                        timeout=TIMEOUT,
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not reach the prediction API at `{API_URL}`: {exc}")
                    st.stop()

            if resp.status_code == 200:
                result = resp.json()
                st.metric("Predicted class", result["class_name"], f"{result['confidence']:.1%} confidence")
                probs = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
                st.markdown("**Class probabilities**")
                for name, p in probs:
                    st.progress(p, text=f"{name} — {p:.1%}")
                api_health.clear()
            elif resp.status_code == 503:
                st.warning(f"Model not available yet. {error_detail(resp)}")
            elif resp.status_code in (415, 422):
                st.error(f"Invalid image: {error_detail(resp)}")
            else:
                st.error(f"API error {resp.status_code}: {error_detail(resp)}")
