from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any
from urllib.parse import urlparse

from ssv_validation.service import SsvValidationError, validate_ssv_workbook


def parse_multipart_form_data(content_type: str, body: bytes) -> dict[str, list[dict[str, Any]]]:
    boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not boundary_match:
        raise ValueError("Missing multipart boundary.")

    boundary = boundary_match.group(1).encode("utf-8")
    delimiter = b"--" + boundary
    fields: dict[str, list[dict[str, Any]]] = {}

    for chunk in body.split(delimiter):
        part = chunk.strip()
        if not part or part == b"--":
            continue

        if part.startswith(b"--"):
            part = part[2:]

        header_blob, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue

        headers = {}
        for line in header_blob.decode("utf-8", "ignore").split("\r\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        disposition = headers.get("content-disposition", "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        if not name_match:
            continue

        name = name_match.group(1)
        payload = content[:-2] if content.endswith(b"\r\n") else content
        fields.setdefault(name, []).append(
            {
                "filename": filename_match.group(1) if filename_match else None,
                "content_type": headers.get("content-type", "application/octet-stream"),
                "data": payload,
            }
        )

    return fields


def is_allowed_blob_url(blob_url: str) -> bool:
    try:
        parsed = urlparse(blob_url)
    except Exception:
        return False

    if parsed.scheme != "https" or not parsed.netloc:
        return False

    hostname = parsed.hostname or ""
    return hostname.endswith(".blob.vercel-storage.com") or hostname.endswith(".public.blob.vercel-storage.com")


def handle_ssv_validation_blob_json(payload: Mapping[str, Any]) -> tuple[HTTPStatus, dict[str, Any]]:
    blob_url = str(payload.get("blobUrl") or "").strip()
    filename = str(payload.get("filename") or "upload.xlsx").strip() or "upload.xlsx"
    include_all_previews = bool(payload.get("includeAllPreviews"))

    if not blob_url:
        return HTTPStatus.BAD_REQUEST, {"success": False, "error": "Missing blob URL."}

    if not is_allowed_blob_url(blob_url):
        return HTTPStatus.BAD_REQUEST, {"success": False, "error": "Invalid blob URL."}

    if not filename.lower().endswith(".xlsx"):
        return HTTPStatus.BAD_REQUEST, {
            "success": False,
            "error": "Invalid file format. Please upload an .xlsx workbook.",
        }

    try:
        with urllib.request.urlopen(blob_url) as response:
            file_bytes = response.read()
    except Exception as exc:
        return HTTPStatus.BAD_REQUEST, {"success": False, "error": f"Unable to download workbook from Blob: {exc}"}

    try:
        response = validate_ssv_workbook(file_bytes, filename, include_all_previews=include_all_previews)
    except SsvValidationError as exc:
        return HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - API safeguard
        return HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": f"Unexpected server error: {exc}"}

    return HTTPStatus.OK, response


def handle_ssv_validation_request(headers: Mapping[str, str], body: bytes) -> tuple[HTTPStatus, dict[str, Any]]:
    try:
        content_type = headers.get("Content-Type") or headers.get("content-type") or ""
    except Exception:
        content_type = ""

    if "application/json" in content_type:
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return HTTPStatus.BAD_REQUEST, {"success": False, "error": "Invalid JSON payload."}

        if not isinstance(payload, dict):
            return HTTPStatus.BAD_REQUEST, {"success": False, "error": "Invalid JSON payload."}

        return handle_ssv_validation_blob_json(payload)

    if "multipart/form-data" not in content_type:
        return HTTPStatus.BAD_REQUEST, {
            "success": False,
            "error": "The upload must use multipart/form-data.",
        }

    try:
        fields = parse_multipart_form_data(content_type, body)
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)}

    uploaded_file = None
    for field_parts in fields.values():
        for part in field_parts:
            if part.get("filename"):
                uploaded_file = part
                break
        if uploaded_file:
            break

    if not uploaded_file:
        return HTTPStatus.BAD_REQUEST, {"success": False, "error": "No Excel file was uploaded."}

    filename = uploaded_file.get("filename") or "upload.xlsx"
    if not filename.lower().endswith(".xlsx"):
        return HTTPStatus.BAD_REQUEST, {
            "success": False,
            "error": "Invalid file format. Please upload an .xlsx workbook.",
        }

    include_all_previews = False
    include_preview_parts = fields.get("includeAllPreviews", [])
    if include_preview_parts:
        include_all_previews = (
            include_preview_parts[0].get("data", b"").decode("utf-8", "ignore").strip().lower() in {"1", "true", "yes", "on"}
        )

    try:
        response = validate_ssv_workbook(uploaded_file["data"], filename, include_all_previews=include_all_previews)
    except SsvValidationError as exc:
        return HTTPStatus.BAD_REQUEST, {"success": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - API safeguard
        return HTTPStatus.INTERNAL_SERVER_ERROR, {"success": False, "error": f"Unexpected server error: {exc}"}

    return HTTPStatus.OK, response
