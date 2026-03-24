from __future__ import annotations

import logging
import os
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ssv_validation.http_api import handle_ssv_validation_request

LOGGER = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parent
class AppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed_path = urlsplit(self.path)
        if parsed_path.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self.path = parsed_path.path
        super().do_GET()

    def do_POST(self) -> None:
        parsed_path = urlsplit(self.path)
        if parsed_path.path not in {"/api", "/api/ssv-validation", "/api/ssv_validation"}:
            self.send_json(HTTPStatus.NOT_FOUND, {"success": False, "error": "Unknown API route."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "Invalid request length."})
            return

        if content_length <= 0:
            self.send_json(HTTPStatus.BAD_REQUEST, {"success": False, "error": "No upload payload was received."})
            return

        body = self.rfile.read(content_length)
        status, response = handle_ssv_validation_request(self.headers, body)
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            LOGGER.exception("Unexpected SSV validation error: %s", response.get("error"))
        elif status >= HTTPStatus.BAD_REQUEST:
            LOGGER.warning("SSV validation failed: %s", response.get("error"))
        self.send_json(status, response)

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppRequestHandler)
    LOGGER.info("Serving 360 view app on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
