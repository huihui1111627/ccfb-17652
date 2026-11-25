"""Stdlib-only web server for the quantization analysis dashboard.

Serves the static single-page app and the results.json produced by
experiments/02_run_analysis.py. No third-party web framework, no network
calls from the page, so it runs fully offline.

    python3 -I tools/run_iso.py -m quant_tool.webapp.server [port]
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
RESULTS = os.path.join(ROOT, "experiments", "results", "results.json")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                return self._send(200, f.read(), _CONTENT_TYPES[".html"])
        if path == "/api/results":
            if not os.path.exists(RESULTS):
                return self._send(404, b'{"error":"results.json missing; '
                                  b'run experiments/02_run_analysis.py"}',
                                  _CONTENT_TYPES[".json"])
            with open(RESULTS, "rb") as f:
                return self._send(200, f.read(), _CONTENT_TYPES[".json"])
        # static assets, confined to the static dir
        rel = path.lstrip("/")
        safe = os.path.normpath(os.path.join(STATIC, rel))
        if not safe.startswith(STATIC) or not os.path.isfile(safe):
            return self._send(404, b"not found", "text/plain")
        ext = os.path.splitext(safe)[1]
        with open(safe, "rb") as f:
            return self._send(200, f.read(),
                              _CONTENT_TYPES.get(ext, "application/octet-stream"))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Quantization dashboard: http://127.0.0.1:{port}")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
