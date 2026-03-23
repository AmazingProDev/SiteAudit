from __future__ import annotations

from http import HTTPStatus

from flask import Flask, Response, request

from ssv_validation.http_api import handle_ssv_validation_request

app = Flask(__name__)


def _cors_json_response(status: HTTPStatus, payload: dict[str, object]) -> Response:
    response = app.response_class(
        response=app.json.dumps(payload),
        status=int(status),
        mimetype="application/json",
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/ssv_validation", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/ssv-validation", methods=["GET", "POST", "OPTIONS"])
def ssv_validation() -> Response:
    if request.method == "OPTIONS":
        return _cors_json_response(HTTPStatus.NO_CONTENT, {})

    if request.method == "GET":
        return _cors_json_response(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"success": False, "error": "Use POST with multipart/form-data."},
        )

    body = request.get_data(cache=False, as_text=False)
    status, payload = handle_ssv_validation_request(request.headers, body)
    return _cors_json_response(status, payload)
