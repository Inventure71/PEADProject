import io
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeHandler:
    def __init__(self):
        self.status = None
        self.headers = []
        self.wfile = io.BytesIO()
        self.ended = False

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers.append((key, value))

    def end_headers(self):
        self.ended = True


def call_get(handler_class):
    fake_handler = FakeHandler()
    handler_class.do_GET(fake_handler)
    body = fake_handler.wfile.getvalue().decode("utf-8")
    return fake_handler, json.loads(body)


class VercelDeploymentTests(unittest.TestCase):
    def test_vercel_api_handlers_return_backend_payloads(self):
        from api.forgetting import handler as ForgettingHandler
        from api.fruit import handler as FruitHandler
        from api.oja import handler as OjaHandler
        from api.playground import handler as PlaygroundHandler

        cases = [
            (FruitHandler, "Fruit Weights", "features"),
            (OjaHandler, "Oja's Rule as Online PCA", "steps"),
            (ForgettingHandler, "Catastrophic Forgetting", "series"),
            (PlaygroundHandler, "Model Playground", "models"),
        ]

        for handler_class, expected_title, required_key in cases:
            with self.subTest(title=expected_title):
                response, payload = call_get(handler_class)

                self.assertEqual(response.status, 200)
                self.assertTrue(response.ended)
                self.assertIn(("Cache-Control", "no-store"), response.headers)
                self.assertEqual(payload["title"], expected_title)
                self.assertIn(required_key, payload)

    def test_vercel_rewrites_point_to_existing_frontend_assets(self):
        config = json.loads((REPO_ROOT / "vercel.json").read_text())
        self.assertIsNone(config["framework"])
        self.assertIn("web-demo/backend/**", config["functions"]["api/*.py"]["includeFiles"])

        destinations = {
            rewrite["destination"].lstrip("/")
            for rewrite in config["rewrites"]
            if rewrite["destination"].startswith("/web-demo/frontend/")
        }

        self.assertEqual(
            destinations,
            {
                "web-demo/frontend/index.html",
                "web-demo/frontend/app.js",
                "web-demo/frontend/styles.css",
            },
        )
        for destination in destinations:
            self.assertTrue((REPO_ROOT / destination).is_file(), destination)


if __name__ == "__main__":
    unittest.main()
