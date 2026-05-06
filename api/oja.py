from __future__ import annotations

from http.server import BaseHTTPRequestHandler

from web_demo_vercel import send_json
from experiments.oja_pca import build_oja_pca_demo


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        send_json(self, build_oja_pca_demo())
