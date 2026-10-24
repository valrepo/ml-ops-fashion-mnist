"""Training-complete webhook notification."""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


class WebhookError(RuntimeError):
    pass


def send_webhook(payload: dict, url: str | None = None, timeout: float = 10.0) -> bool:
    """POST ``payload`` as JSON to ``url`` (default: ``$WEBHOOK_URL``).

    Returns ``False`` if no URL is configured; raises :class:`WebhookError` if delivery fails.
    """
    url = url if url is not None else os.environ.get("WEBHOOK_URL", "")
    if not url:
        log.info("WEBHOOK_URL not set - skipping notification")
        return False
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise WebhookError(f"Webhook delivery to {url} failed: {exc}") from exc
    log.info("Webhook %s delivered to %s (HTTP %s)", payload.get("event"), url, resp.status_code)
    return True
