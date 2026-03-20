from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

from .analyzer import SsvAnalysisError, analyze_bitmap
from .imaging import SsvImageError, convert_image_to_bmp, decode_bmp
from .kpi_analyzer import SsvKpiError, analyze_kpi_bitmap
from .legend_mapping import enrich_warning_messages
from .throughput import evaluate_avg_throughput
from .workbook import SsvWorkbookError, extract_avg_throughput_metrics, row_col_to_ref, select_target_images

LOGGER = logging.getLogger(__name__)


class SsvValidationError(ValueError):
    """Top-level validation error surfaced by the API."""


def validate_ssv_workbook(file_bytes: bytes, filename: str) -> dict[str, object]:
    try:
        selected_images = select_target_images(file_bytes)
    except SsvWorkbookError as exc:
        raise SsvValidationError(str(exc)) from exc

    analyses = []
    for candidate, image_bytes, mime_type in selected_images:
        preview_image = "data:{};base64,{}".format(
            mime_type,
            base64.b64encode(image_bytes).decode("ascii"),
        )

        suffix = Path(candidate.media_path).suffix or ".img"
        with tempfile.TemporaryDirectory(prefix="ssv-validation-") as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / f"ssv-source{suffix}"
            bitmap_path = temp_path / "ssv-source.bmp"
            source_path.write_bytes(image_bytes)

            try:
                convert_image_to_bmp(source_path, bitmap_path)
                bitmap = decode_bmp(bitmap_path)
                if candidate.analysis_kind == "degradation":
                    outcome = analyze_kpi_bitmap(
                        bitmap=bitmap,
                        preview_image_uri=preview_image,
                        metric_name=candidate.metric_name,
                        metric_group=candidate.metric_group,
                    )
                else:
                    outcome = analyze_bitmap(bitmap, preview_image)
            except (SsvImageError, SsvAnalysisError, SsvKpiError) as exc:
                raise SsvValidationError(f"{candidate.target_label}: {exc}") from exc

            warnings = outcome.warnings
            if candidate.analysis_kind != "degradation":
                try:
                    warnings = enrich_warning_messages(
                        file_bytes=file_bytes,
                        candidate=candidate,
                        bitmap=bitmap,
                        detected_colors=outcome.detected_colors,
                        warning_details=outcome.warning_details,
                    )
                except Exception as exc:  # pragma: no cover - best effort enrichment
                    LOGGER.warning("Legend warning enrichment failed for %s: %s", candidate.sheet_name, exc)
                    warnings = outcome.warnings

        is_failure = outcome.is_failure if outcome.is_failure is not None else outcome.cross
        analyses.append(
            {
                "label": candidate.target_label,
                "targetKey": candidate.target_key,
                "analysisKind": candidate.analysis_kind,
                "cross": outcome.cross,
                "isFailure": is_failure,
                "verdict": outcome.verdict,
                "warnings": warnings,
                "detected_colors": [
                    {
                        "name": detected_color.name,
                        "rgb": list(detected_color.rgb),
                        "hex": detected_color.hex,
                        "dominant_angle": round(detected_color.dominant_angle, 2),
                        "site_angle": round(detected_color.site_angle, 2) if detected_color.site_angle is not None else None,
                        "point_count": detected_color.point_count,
                    }
                    for detected_color in outcome.detected_colors
                ],
                "metrics": outcome.metrics,
                "siteCenter": outcome.site_center,
                "previewImage": preview_image,
                "annotatedPreview": outcome.annotated_preview,
                "selection": {
                    "sheetName": candidate.sheet_name,
                    "sheetPath": candidate.sheet_path,
                    "drawingPath": candidate.drawing_path,
                    "mediaPath": candidate.media_path,
                    "metricGroup": candidate.metric_group,
                    "metricName": candidate.metric_name,
                    "anchor": {
                        "row": candidate.anchor_row + 1,
                        "col": candidate.anchor_col + 1,
                        "ref": row_col_to_ref(candidate.anchor_row, candidate.anchor_col),
                    },
                    "captionCell": candidate.caption_ref,
                    "nearbyText": candidate.nearby_text,
                },
            }
        )

    throughput_metrics = extract_avg_throughput_metrics(file_bytes, filename)
    if throughput_metrics is not None:
        throughput_analysis = evaluate_avg_throughput(throughput_metrics)
        throughput_analysis["selection"] = {
            "sheetName": throughput_metrics["sheet_name"],
            "sheetPath": None,
            "drawingPath": None,
            "mediaPath": None,
            "metricGroup": "throughput",
            "metricName": "Avg Throughput",
            "anchor": None,
            "captionCell": throughput_metrics.get("dl_label_ref"),
            "nearbyText": [
                f"{throughput_metrics['ul_label_ref']}: Avg UL Throughput (Mbps)",
                f"{throughput_metrics['ul_value_ref']}: {throughput_metrics['ul_value_mbps']:.2f} Mbps",
                f"{throughput_metrics['dl_label_ref']}: Avg DL Throughput (Mbps)",
                f"{throughput_metrics['dl_value_ref']}: {throughput_metrics['dl_value_mbps']:.2f} Mbps",
            ],
        }
        analyses.append(throughput_analysis)

    if not analyses:
        raise SsvValidationError("No analyzable SSV images were extracted from the workbook.")

    primary = analyses[0]
    overall_failure = any(bool(analysis.get("isFailure")) for analysis in analyses)
    has_ssv_checks = any(analysis.get("analysisKind") in {"degradation", "throughput_average"} for analysis in analyses)
    overall_cross = any(analysis["cross"] for analysis in analyses)
    if any(bool(analysis.get("isFailure")) and analysis.get("analysisKind") in {"degradation", "throughput_average"} for analysis in analyses):
        overall_verdict = "SSV NOK"
    elif overall_cross:
        overall_verdict = "Cross detected"
    elif has_ssv_checks:
        overall_verdict = "SSV OK"
    else:
        overall_verdict = "No cross detected"

    return {
        "success": True,
        "isFailure": overall_failure,
        "cross": overall_cross,
        "verdict": overall_verdict,
        "analysisCount": len(analyses),
        "analyses": analyses,
        "summary": {
            "crossedImages": sum(1 for analysis in analyses if analysis["cross"]),
            "failedImages": sum(1 for analysis in analyses if analysis.get("isFailure")),
            "clearImages": sum(1 for analysis in analyses if not analysis.get("isFailure")),
            "labels": [analysis["label"] for analysis in analyses],
            "warnings": [warning for analysis in analyses for warning in analysis.get("warnings", [])],
        },
        "detected_colors": primary["detected_colors"],
        "metrics": primary["metrics"],
        "siteCenter": primary["siteCenter"],
        "previewImage": primary["previewImage"],
        "annotatedPreview": primary["annotatedPreview"],
        "selection": primary["selection"],
        "warnings": primary.get("warnings", []),
    }
