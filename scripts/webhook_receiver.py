"""Minimal local webhook receiver for testing training notifications.

    python scripts/webhook_receiver.py            # listens on :8088
    POST /webhook  -> logs the JSON payload, appends it to $WEBHOOK_LOG_DIR/payloads.jsonl
    GET  /webhook  -> returns the payloads received so far
"""

import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("PORT", "8088"))
LOG_DIR = Path(os.environ.get("WEBHOOK_LOG_DIR", "webhook_payloads"))
RECEIVED: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/webhook", ""):
            return self._reply(200, {"count": len(RECEIVED), "payloads": RECEIVED})
        if self.path == "/health":
            return self._reply(200, {"status": "ok"})
        self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/webhook":
            return self._reply(404, {"error": "not found"})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return self._reply(400, {"error": "invalid JSON"})
        entry = {"received_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
        RECEIVED.append(entry)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "payloads.jsonl").open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(f"== webhook received: {payload.get('event')}\n{json.dumps(payload, indent=2)}", flush=True)
        self._reply(200, {"ok": True})

    def log_message(self, fmt, *args):  # keep the console readable
        pass


if __name__ == "__main__":
    print(f"Webhook receiver listening on :{PORT} (POST /webhook)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
