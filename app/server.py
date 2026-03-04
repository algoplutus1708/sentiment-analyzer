import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.model import SentimentPredictor


BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

predictor = SentimentPredictor(checkpoint_path=str(BASE_DIR / "tinyllm_complete.pt"))


class AppHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type="text/plain; charset=utf-8"):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")

        if parsed.path == "/health":
            return self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model_loaded": True,
                    "labels": predictor.class_names,
                    "inference_version": predictor.inference_version,
                },
            )

        if parsed.path.startswith("/static/"):
            relative = parsed.path.replace("/static/", "", 1)
            file_path = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in file_path.parents and file_path != STATIC_DIR.resolve():
                return self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")

            if file_path.suffix == ".css":
                return self._send_file(file_path, "text/css; charset=utf-8")
            if file_path.suffix == ".js":
                return self._send_file(file_path, "application/javascript; charset=utf-8")
            return self._send_file(file_path, "application/octet-stream")

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            return self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Invalid JSON payload"})

        text = str(payload.get("text", "")).strip()
        if not text:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Text is required"})

        try:
            prediction = predictor.predict(text)
        except Exception as exc:
            return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": f"Prediction failed: {exc}"})

        return self._send_json(
            HTTPStatus.OK,
            {
                "label": prediction.label,
                "confidence": round(prediction.confidence, 6),
                "probabilities": {k: round(v, 6) for k, v in prediction.probabilities.items()},
                "inference_version": predictor.inference_version,
            },
        )


def run(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
