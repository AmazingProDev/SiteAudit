from __future__ import annotations

import base64
import math

from .analyzer import (
    COLOR_SATURATION_THRESHOLD,
    COLOR_VALUE_THRESHOLD,
    LEGEND_X_RATIO,
    LEGEND_Y_RATIO,
    extract_components,
    is_point_like_component,
    rgb_to_hex,
    rgb_to_hsv,
)
from .models import AnalysisOutcome, Bitmap

RED_HUE_WINDOW_DEG = 14.0
RED_POINT_RATIO_THRESHOLD = 0.10
MIN_CONTINUOUS_RED_POINTS = 6
MIN_TOTAL_KPI_POINTS = 20
CLUSTER_LINK_DISTANCE = 18.0
HOTSPOT_PADDING = 18.0


class SsvKpiError(ValueError):
    """Raised when a KPI map cannot be analyzed reliably."""


def analyze_kpi_bitmap(
    bitmap: Bitmap,
    preview_image_uri: str,
    metric_name: str | None,
    metric_group: str | None,
) -> AnalysisOutcome:
    point_components = extract_kpi_point_components(bitmap)
    total_points = len(point_components)
    if total_points < MIN_TOTAL_KPI_POINTS:
        raise SsvKpiError("The extracted KPI image does not contain enough colored measurement points for degradation analysis.")

    red_components = [component for component in point_components if is_red_component(bitmap, component)]
    red_point_count = len(red_components)
    red_point_ratio = (red_point_count / total_points) if total_points else 0.0
    red_clusters = cluster_components(red_components)
    largest_cluster = max(red_clusters, key=len, default=[])
    continuous_red_count = len(largest_cluster)
    use_red_ratio = metric_group in {"coverage", "quality"}
    degradation_detected = continuous_red_count >= MIN_CONTINUOUS_RED_POINTS
    if use_red_ratio and red_point_ratio > RED_POINT_RATIO_THRESHOLD:
        degradation_detected = True

    warnings: list[str] = []
    if continuous_red_count >= MIN_CONTINUOUS_RED_POINTS:
        warnings.append(f"Continuous red points detected ({continuous_red_count}).")
    if use_red_ratio and red_point_ratio > RED_POINT_RATIO_THRESHOLD:
        warnings.append(f"Red point ratio {red_point_ratio * 100.0:.1f}% exceeds 10%.")

    verdict = "SSV NOK" if degradation_detected else "SSV OK"
    hotspot = largest_cluster or red_components
    annotated_preview = build_kpi_annotated_preview(
        bitmap=bitmap,
        preview_image_uri=preview_image_uri,
        verdict=verdict,
        metric_name=metric_name,
        hotspot_components=hotspot,
        show_hotspot=degradation_detected and bool(hotspot),
    )

    metrics = {
        "metric_name": metric_name or "KPI",
        "metric_group": metric_group or "kpi",
        "total_point_count": total_points,
        "red_point_count": red_point_count,
        "red_point_ratio": round(red_point_ratio, 4),
        "continuous_red_count": continuous_red_count,
        "degradation_detected": degradation_detected,
        "warnings": warnings,
    }

    return AnalysisOutcome(
        cross=False,
        verdict=verdict,
        detected_colors=[],
        metrics=metrics,
        site_center={"x": 0.0, "y": 0.0},
        annotated_preview=annotated_preview,
        analysis_kind="degradation",
        is_failure=degradation_detected,
        warnings=warnings,
        warning_details=[],
    )


def extract_kpi_point_components(bitmap: Bitmap) -> list[dict[str, object]]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    mask = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if x < legend_x and y < legend_y:
                continue
            red, green, blue = bitmap.pixels[y][x]
            _hue, saturation, value = rgb_to_hsv(red, green, blue)
            if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                mask[y][x] = 1

    return [component for component in extract_components(mask) if is_point_like_component(component)]


def is_red_component(bitmap: Bitmap, component: dict[str, object]) -> bool:
    samples = [bitmap.pixels[y][x] for x, y in component["pixels"]]
    red = sum(sample[0] for sample in samples) / len(samples)
    green = sum(sample[1] for sample in samples) / len(samples)
    blue = sum(sample[2] for sample in samples) / len(samples)
    hue, saturation, value = rgb_to_hsv(int(red), int(green), int(blue))

    if saturation < COLOR_SATURATION_THRESHOLD or value < COLOR_VALUE_THRESHOLD:
        return False

    hue_degrees = hue * 360.0
    return hue_degrees <= RED_HUE_WINDOW_DEG or hue_degrees >= (360.0 - RED_HUE_WINDOW_DEG)


def cluster_components(components: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    if not components:
        return []

    centers = [component_center(component) for component in components]
    visited = [False] * len(components)
    clusters: list[list[dict[str, object]]] = []

    for index in range(len(components)):
        if visited[index]:
            continue

        stack = [index]
        visited[index] = True
        cluster: list[dict[str, object]] = []

        while stack:
            current = stack.pop()
            cluster.append(components[current])
            current_x, current_y = centers[current]

            for other_index in range(len(components)):
                if visited[other_index]:
                    continue
                other_x, other_y = centers[other_index]
                if math.hypot(current_x - other_x, current_y - other_y) > CLUSTER_LINK_DISTANCE:
                    continue
                visited[other_index] = True
                stack.append(other_index)

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


def component_center(component: dict[str, object]) -> tuple[float, float]:
    xs = [pixel[0] for pixel in component["pixels"]]
    ys = [pixel[1] for pixel in component["pixels"]]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def hotspot_circle(components: list[dict[str, object]]) -> tuple[float, float, float] | None:
    if not components:
        return None

    pixels = [pixel for component in components for pixel in component["pixels"]]
    xs = [pixel[0] for pixel in pixels]
    ys = [pixel[1] for pixel in pixels]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    radius = max(math.hypot(x - center_x, y - center_y) for x, y in pixels) + HOTSPOT_PADDING
    return center_x, center_y, radius


def build_kpi_annotated_preview(
    bitmap: Bitmap,
    preview_image_uri: str,
    verdict: str,
    metric_name: str | None,
    hotspot_components: list[dict[str, object]],
    show_hotspot: bool,
) -> str:
    width = bitmap.width
    height = bitmap.height
    circle_markup = ""
    if show_hotspot:
        circle = hotspot_circle(hotspot_components)
        if circle is not None:
            center_x, center_y, radius = circle
            circle_markup = (
                f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{radius:.2f}" '
                'fill="none" stroke="#ff4d4f" stroke-width="4" stroke-dasharray="8 6" />'
            )

    metric_label = metric_name or "KPI"
    verdict_bg = "#431217" if verdict == "SSV NOK" else "#10331b"
    verdict_fg = "#ffd9db" if verdict == "SSV NOK" else "#c8ffd7"
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="{preview_image_uri}" width="{width}" height="{height}" />
  <rect x="{width - 170}" y="14" width="156" height="64" rx="12" fill="{verdict_bg}" opacity="0.92" />
  <text x="{width - 156}" y="40" fill="{verdict_fg}" font-size="18" font-weight="700" font-family="Inter, sans-serif">{verdict}</text>
  <text x="{width - 156}" y="61" fill="#d7e8ff" font-size="12" font-family="Inter, sans-serif">{metric_label}</text>
  {circle_markup}
</svg>
""".strip()

    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
