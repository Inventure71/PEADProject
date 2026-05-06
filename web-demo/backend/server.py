from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from experiments.forgetting import build_forgetting_demo
from experiments.fruit import build_fruit_demo, build_playground_demo
from experiments.oja_pca import build_oja_pca_demo


BACKEND_ROOT = Path(__file__).resolve().parent
WEB_DEMO_ROOT = BACKEND_ROOT.parent
FRONTEND_ROOT = WEB_DEMO_ROOT / "frontend"


def _json_response(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class DemoRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/fruit":
            _json_response(self, build_fruit_demo())
            return
        if path == "/api/oja":
            _json_response(self, build_oja_pca_demo())
            return
        if path == "/api/forgetting":
            _json_response(self, build_forgetting_demo())
            return
        if path == "/api/playground":
            _json_response(self, build_playground_demo())
            return

        self._serve_static(path)

    def log_message(self, format: str, *args) -> None:
        print("%s - - %s" % (self.address_string(), format % args))

    def _serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"

        requested = (FRONTEND_ROOT / path.lstrip("/")).resolve()
        if FRONTEND_ROOT.resolve() not in requested.parents and requested != FRONTEND_ROOT.resolve():
            self.send_error(403)
            return

        if not requested.exists() or not requested.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        body = requested.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    port = int(os.environ.get("PORT", "5173"))
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoRequestHandler)
    print(f"PEAD web demo running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
