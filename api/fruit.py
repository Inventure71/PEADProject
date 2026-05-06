from __future__ import annotations

from http.server import BaseHTTPRequestHandler

from web_demo_vercel import send_json
from experiments.fruit import build_fruit_demo


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        send_json(self, build_fruit_demo())
