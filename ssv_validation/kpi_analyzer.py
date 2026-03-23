from __future__ import annotations

import base64
import math
import time
import statistics
from collections.abc import Sequence
from typing import Any

from .acceleration import bitmap_hsv_arrays, bitmap_rgb_array, cv2, extract_binary_components, np, rgb_pixel
from .analyzer import (
    COLOR_SATURATION_THRESHOLD,
    COLOR_VALUE_THRESHOLD,
    LEGEND_X_RATIO,
    LEGEND_Y_RATIO,
    rgb_to_hex,
    rgb_to_hsv,
)
from .models import AnalysisOutcome, Bitmap, DotComponent, LegendSwatch

RED_HUE_WINDOW_DEG = 35.0
LEGEND_HUE_WINDOW_DEG = 10.0
LEGEND_DISTANCE_MARGIN_RATIO = 0.82
CLAUDE_PROJECT_MIN_DOT_COUNT = 5
MIN_CONTINUOUS_RED_POINTS = 6
MIN_SPARSE_KPI_POINTS = 6
MIN_TOTAL_KPI_POINTS = 20
CLUSTER_LINK_DISTANCE = 18.0
RED_RUN_LINK_DISTANCE = 84.0
RED_POINT_GAP_DISTANCE = 115.0
ADAPTIVE_LINK_DISTANCE_MULTIPLIER = 1.6
ADAPTIVE_LINK_DISTANCE_MIN = 18.0
ADAPTIVE_LINK_DISTANCE_MAX = 96.0
DEGRADED_COMPONENT_CLUSTER_MULTIPLIER = 1.8
DEGRADED_COMPONENT_CLUSTER_MIN = 24.0
DEGRADED_COMPONENT_CLUSTER_MAX = 320.0
DIRECT_OVERLAY_CLUSTER_MULTIPLIER = 6.8
DIRECT_OVERLAY_CLUSTER_MIN = 180.0
DIRECT_OVERLAY_CLUSTER_MAX = 320.0
DOT_CHAIN_LINK_DISTANCE_MULTIPLIER = 1.85
DOT_CHAIN_LINK_DISTANCE_MIN = 18.0
DOT_CHAIN_LINK_DISTANCE_MAX = 120.0
DOT_CHAIN_NEIGHBOR_RANK_LIMIT = 3
DOT_CHAIN_MIN_PAIR_ANGLE_DEG = 140.0
DOT_CHAIN_MERGE_DISTANCE_MULTIPLIER = 1.75
DOT_CHAIN_MERGE_DISTANCE_MAX = 210.0
DOT_CHAIN_MERGE_ALIGNMENT_MIN = 0.3
DOT_CHAIN_MERGE_ALIGNMENT_GOOD = 0.6
HOTSPOT_PADDING = 18.0
RED_RUN_COMPONENT_MIN_PIXELS = 20
RED_RUN_COMPONENT_MAX_PIXELS = 2200
RED_RUN_COMPONENT_MIN_SPAN = 5
RED_RUN_COMPONENT_MAX_SPAN = 48
RED_RUN_COMPONENT_MIN_FILL = 0.35
RED_RUN_COMPONENT_ASPECT_MIN = 0.55
RED_RUN_COMPONENT_ASPECT_MAX = 1.8
RED_RUN_DOMINANT_AREA_RATIO_MIN = 0.45
RED_RUN_DOMINANT_AREA_RATIO_MAX = 1.85
INNER_WHITE_SYMBOL_MIN_PIXELS = 20
OVERLAY_POINT_ASPECT_MIN = 0.72
OVERLAY_POINT_ASPECT_MAX = 1.38
OVERLAY_POINT_FILL_MIN = 0.42
OVERLAY_POINT_FILL_MAX = 0.95
OVERLAY_POINT_CENTER_SOLID_MIN_RATIO = 0.62
OVERLAY_POINT_CENTER_WHITE_MAX_RATIO = 0.18
KPI_COMPONENT_MIN_PIXELS = 20
KPI_COMPONENT_MAX_PIXELS = 2200
KPI_COMPONENT_MIN_SPAN = 5
KPI_COMPONENT_MAX_SPAN = 48
KPI_COMPONENT_MIN_FILL = 0.28
KPI_COMPONENT_ASPECT_MIN = 0.45
KPI_COMPONENT_ASPECT_MAX = 2.2
KPI_COMPONENT_DOMINANT_AREA_RATIO_MIN = 0.35
KPI_COMPONENT_DOMINANT_AREA_RATIO_MAX = 2.1


class SsvKpiError(ValueError):
    """Raised when a KPI map cannot be analyzed reliably."""


def analyze_kpi_bitmap(
    bitmap: Bitmap,
    preview_image_uri: str,
    metric_name: str | None,
    metric_group: str | None,
    sheet_name: str | None = None,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> AnalysisOutcome:
    stage_started = time.perf_counter()
    point_components = extract_kpi_point_components(bitmap)
    point_extraction_s = time.perf_counter() - stage_started
    total_points = len(point_components)
    if total_points < MIN_SPARSE_KPI_POINTS:
        raise SsvKpiError("The extracted KPI image does not contain enough colored measurement points for degradation analysis.")

    warnings: list[str] = []
    stage_started = time.perf_counter()
    red_component_indexes = extract_visual_red_dot_indexes(
        bitmap,
        point_components,
        legend_swatches_override=legend_swatches_override,
        degraded_swatch_override=degraded_swatch_override,
    )
    degraded_classification_s = time.perf_counter() - stage_started
    red_components = [point_components[index] for index in red_component_indexes]
    stage_started = time.perf_counter()
    chain_link_distance = estimate_dot_chain_link_distance(point_components)
    dot_chain_indexes = build_ordered_dot_chain_indexes(point_components, chain_link_distance)
    chain_build_s = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    qualifying_run_indexes = extract_qualifying_degraded_run_indexes(
        dot_chain_indexes,
        point_components,
        red_component_indexes,
        chain_link_distance,
    )
    run_extraction_s = time.perf_counter() - stage_started
    continuity_strategy = "ordered_chain"
    red_point_count = len(red_components)
    red_point_ratio = (red_point_count / total_points) if total_points else 0.0
    highlighted_clusters: list[list[dict[str, object] | DotComponent]] = []
    hotspot_circles: list[tuple[float, float, float]] = []
    continuous_red_count = 0

    stage_started = time.perf_counter()
    fallback_red_components: list[DotComponent] = []
    metric_group_name = (metric_group or "").lower()
    sheet_name_value = (sheet_name or "").lower()
    exact_red_cluster_summaries = build_exact_red_cluster_summaries(
        bitmap,
        min_dot_count=CLAUDE_PROJECT_MIN_DOT_COUNT,
    )
    if exact_red_cluster_summaries:
        run_summaries = exact_red_cluster_summaries
        continuity_strategy = "claude_ssv_red_cluster"
        red_point_count = sum(int(summary.get("dot_count", 0)) for summary in run_summaries)
        red_point_ratio = (red_point_count / total_points) if total_points else 0.0
    else:
        run_summaries = []
    if run_summaries:
        run_summaries.sort(
            key=lambda summary: (
                -float(summary.get("route_score", 0.0)),
                -len(summary["indexes"]),
                -summary["total_area"],
                -summary["max_extent"],
                summary["sort_key"][0],
                summary["sort_key"][1],
            ),
        )
        highlighted_clusters = [summary["components"] for summary in run_summaries]
        hotspot_circles = [
            circle
            for summary in run_summaries
            for circle in summary.get("circles", ([] if summary.get("circle") is None else [summary["circle"]]))
            if circle is not None
        ]
        continuous_red_count = int(run_summaries[0].get("dot_count", len(highlighted_clusters[0])))
    summary_sort_s = time.perf_counter() - stage_started

    degradation_detected = bool(hotspot_circles)

    if continuous_red_count >= MIN_CONTINUOUS_RED_POINTS:
        warnings.append(f"Continuous red points detected ({continuous_red_count}).")

    verdict = "SSV NOK" if degradation_detected else "SSV OK"
    stage_started = time.perf_counter()
    annotated_preview = (
        build_kpi_annotated_preview(
            bitmap=bitmap,
            preview_image_uri=preview_image_uri,
            hotspot_circles=hotspot_circles,
        )
        if degradation_detected and hotspot_circles
        else ""
    )
    annotation_s = time.perf_counter() - stage_started

    metrics = {
        "metric_name": metric_name or "KPI",
        "metric_group": metric_group or "kpi",
        "total_point_count": total_points,
        "red_point_count": red_point_count,
        "red_point_ratio": round(red_point_ratio, 4),
        "continuous_red_count": continuous_red_count,
        "degradation_run_count": len(highlighted_clusters),
        "red_link_distance": round(chain_link_distance, 2),
        "red_cluster_strategy": continuity_strategy,
        "degradation_detected": degradation_detected,
        "warnings": warnings,
        "stage_timings": {
            "point_extraction_s": round(point_extraction_s, 4),
            "degraded_classification_s": round(degraded_classification_s, 4),
            "chain_build_s": round(chain_build_s, 4),
            "run_extraction_s": round(run_extraction_s, 4),
            "summary_sort_s": round(summary_sort_s, 4),
            "annotation_s": round(annotation_s, 4),
        },
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


def extract_kpi_point_components(bitmap: Bitmap) -> list[DotComponent]:
    return extract_candidate_dot_components(bitmap, exclude_legend=True)


def extract_candidate_dot_components(
    bitmap: Bitmap,
    exclude_legend: bool,
) -> list[DotComponent]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    hsv_arrays = bitmap_hsv_arrays(bitmap)
    if hsv_arrays is not None:
        _hue, saturation, value = hsv_arrays
        mask = (saturation >= COLOR_SATURATION_THRESHOLD) & (value >= COLOR_VALUE_THRESHOLD)
        if exclude_legend:
            mask = mask.copy()
            mask[:legend_y, :legend_x] = False
        mask_array = mask.astype(np.uint8)
        mask_rows = None
    else:
        mask_rows = [[0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if exclude_legend and x < legend_x and y < legend_y:
                    continue
                red, green, blue = rgb_pixel(bitmap, x, y)
                _hue, saturation, value = rgb_to_hsv(red, green, blue)
                if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                    mask_rows[y][x] = 1
        mask_array = None

    if mask_array is not None and np is not None and cv2 is not None:
        accelerated_components = extract_candidate_dot_components_accelerated(bitmap, mask_array)
        if accelerated_components:
            return accelerated_components

    raw_components = extract_binary_components(mask_rows=mask_rows, mask_array=mask_array)
    raw_components = [component for component in raw_components if is_kpi_measurement_component_stats(component)]
    if not raw_components:
        return []

    dominant_area = estimate_dominant_component_area(raw_components)
    min_area = max(KPI_COMPONENT_MIN_PIXELS, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MIN)
    max_area = max(min_area, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MAX)
    filtered = [component for component in raw_components if min_area <= int(component["area"]) <= max_area]
    selected = filtered or raw_components
    return [build_dot_component(bitmap, component) for component in selected]


def should_use_exact_red_cluster_method(
    metric_name: str | None,
    metric_group_name: str,
    sheet_name_value: str,
) -> bool:
    metric_name_value = (metric_name or "").strip().lower()
    if metric_group_name == "throughput":
        return True

    if metric_name_value not in {"sinr", "rsrp"}:
        return False

    if "volte" in sheet_name_value:
        return True

    return any(token in sheet_name_value for token in ("l800", "l2100", "l2600", "lte"))


def extract_candidate_dot_components_accelerated(
    bitmap: Bitmap,
    mask_array: Any,
) -> list[DotComponent]:
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_array.astype(np.uint8, copy=False), connectivity=8)
    if labels_count <= 1:
        return []

    raw_components: list[dict[str, object]] = []
    for label_index in range(1, labels_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        left = int(stats[label_index, cv2.CC_STAT_LEFT])
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        component = {
            "label": label_index,
            "area": area,
            "width": width,
            "height": height,
            "bbox": (left, top, left + width - 1, top + height - 1),
            "center": (float(centroids[label_index][0]), float(centroids[label_index][1])),
        }
        if is_kpi_measurement_component_stats(component):
            raw_components.append(component)

    if not raw_components:
        return []

    dominant_area = estimate_dominant_component_area(raw_components)
    min_area = max(KPI_COMPONENT_MIN_PIXELS, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MIN)
    max_area = max(min_area, dominant_area * KPI_COMPONENT_DOMINANT_AREA_RATIO_MAX)
    selected = [component for component in raw_components if min_area <= int(component["area"]) <= max_area] or raw_components

    rgb_array = bitmap_rgb_array(bitmap)
    materialized: list[DotComponent] = []
    for component in selected:
        materialized.append(build_accelerated_dot_component(bitmap, component, labels, rgb_array))
    return materialized


def build_accelerated_dot_component(
    bitmap: Bitmap,
    component: dict[str, object],
    labels: Any,
    rgb_array: Any,
) -> DotComponent:
    label_index = int(component["label"])
    min_x, min_y, max_x, max_y = component["bbox"]
    width = int(component["width"])
    height = int(component["height"])
    area = int(component["area"])

    label_roi = labels[min_y : max_y + 1, min_x : max_x + 1]
    local_mask = label_roi == label_index
    local_ys, local_xs = np.where(local_mask)
    xs = local_xs.astype(np.int32) + int(min_x)
    ys = local_ys.astype(np.int32) + int(min_y)

    if rgb_array is not None:
        roi_rgb = rgb_array[min_y : max_y + 1, min_x : max_x + 1]
        component_rgb = roi_rgb[local_mask]
        red_green_blue = component_rgb.mean(axis=0)
        mean_red = float(red_green_blue[0])
        mean_green = float(red_green_blue[1])
        mean_blue = float(red_green_blue[2])
    else:
        mean_red = sum(rgb_pixel(bitmap, int(x), int(y))[0] for x, y in zip(xs.tolist(), ys.tolist())) / area
        mean_green = sum(rgb_pixel(bitmap, int(x), int(y))[1] for x, y in zip(xs.tolist(), ys.tolist())) / area
        mean_blue = sum(rgb_pixel(bitmap, int(x), int(y))[2] for x, y in zip(xs.tolist(), ys.tolist())) / area

    fill_ratio = area / float(max(1, width * height))
    center = (float(component["center"][0]), float(component["center"][1]))
    mean_lab = rgb_to_lab(mean_red, mean_green, mean_blue)
    pixels = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
    return DotComponent(
        pixels=pixels,
        area=area,
        bbox=(int(min_x), int(min_y), int(max_x), int(max_y)),
        center=center,
        width=width,
        height=height,
        fill_ratio=fill_ratio,
        mean_rgb=(mean_red, mean_green, mean_blue),
        mean_lab=mean_lab,
    )


def build_dot_component(bitmap: Bitmap, component: dict[str, object] | DotComponent) -> DotComponent:
    if isinstance(component, DotComponent):
        return component

    pixels = component["pixels"]
    bbox = component.get("bbox")
    area = int(component.get("area", len(pixels)))
    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        width = int(component.get("width", max_x - min_x + 1))
        height = int(component.get("height", max_y - min_y + 1))
    else:
        xs = [pixel[0] for pixel in pixels]
        ys = [pixel[1] for pixel in pixels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x + 1
        height = max_y - min_y + 1

    fill_ratio = area / float(max(1, width * height))
    xs_array = component.get("xs")
    ys_array = component.get("ys")
    rgb_array = bitmap_rgb_array(bitmap)
    if rgb_array is not None and xs_array is not None and ys_array is not None and len(xs_array) == area:
        red_green_blue = rgb_array[ys_array, xs_array].mean(axis=0)
        mean_red = float(red_green_blue[0])
        mean_green = float(red_green_blue[1])
        mean_blue = float(red_green_blue[2])
        center = (float(xs_array.mean()), float(ys_array.mean()))
    else:
        mean_red = sum(rgb_pixel(bitmap, x, y)[0] for x, y in pixels) / area
        mean_green = sum(rgb_pixel(bitmap, x, y)[1] for x, y in pixels) / area
        mean_blue = sum(rgb_pixel(bitmap, x, y)[2] for x, y in pixels) / area
        center = (
            sum(pixel[0] for pixel in pixels) / area,
            sum(pixel[1] for pixel in pixels) / area,
        )
    mean_lab = rgb_to_lab(mean_red, mean_green, mean_blue)
    return DotComponent(
        pixels=pixels,
        area=area,
        bbox=(min_x, min_y, max_x, max_y),
        center=center,
        width=width,
        height=height,
        fill_ratio=fill_ratio,
        mean_rgb=(mean_red, mean_green, mean_blue),
        mean_lab=mean_lab,
    )


def extract_direct_degraded_components(
    bitmap: Bitmap,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> list[DotComponent]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    legend_swatches = list(legend_swatches_override) if legend_swatches_override is not None else detect_legend_swatches(bitmap, legend_x, legend_y)
    degraded_swatch = degraded_swatch_override if degraded_swatch_override is not None else resolve_degraded_swatch(legend_swatches)

    hsv_arrays = bitmap_hsv_arrays(bitmap)
    rgb_array = bitmap_rgb_array(bitmap)
    if hsv_arrays is not None and rgb_array is not None and np is not None:
        _hue, saturation, value = hsv_arrays
        red = rgb_array[:, :, 0].astype(np.float32, copy=False)
        green = rgb_array[:, :, 1].astype(np.float32, copy=False)
        blue = rgb_array[:, :, 2].astype(np.float32, copy=False)
        mask_array = (
            (saturation >= 0.35)
            & (value >= 0.30)
            & (red >= (green * 1.05))
            & (red >= (blue * 1.15))
        ).astype(np.uint8)
        mask_array[:legend_y, :legend_x] = 0
        raw_components = extract_binary_components(mask_array=mask_array)
    else:
        mask_rows = [[0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                if x < legend_x and y < legend_y:
                    continue
                red, green, blue = rgb_pixel(bitmap, x, y)
                hue, saturation, value = rgb_to_hsv(red, green, blue)
                if saturation >= 0.35 and value >= 0.30 and red >= (green * 1.05) and red >= (blue * 1.15):
                    mask_rows[y][x] = 1
        raw_components = extract_binary_components(mask_rows=mask_rows)

    components: list[DotComponent] = []
    for raw_component in raw_components:
        component = build_dot_component(bitmap, raw_component)
        if component.area < 20 or component.area > 2500:
            continue
        if component.width < 4 or component.height < 4 or component.width > 80 or component.height > 80:
            continue
        hue, _saturation, _value = rgb_to_hsv(int(component.mean_rgb[0]), int(component.mean_rgb[1]), int(component.mean_rgb[2]))
        if not is_degraded_component_color(component, degraded_swatch, legend_swatches, hue * 360.0):
            continue
        if not is_solid_circle_like_component(bitmap, component):
            continue
        components.append(component)
    return components


def build_exact_red_cluster_summaries(
    bitmap: Bitmap,
    min_dot_count: int,
    padding: int = 25,
) -> list[dict[str, object]]:
    if cv2 is None or np is None:
        return []

    rgb_array = bitmap_rgb_array(bitmap)
    if rgb_array is None:
        return []

    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    h_img, w_img = bgr_image.shape[:2]
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 150, 150], dtype=np.uint8)
    upper_red1 = np.array([8, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([172, 150, 150], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    legend_w = int(w_img * 0.22)
    legend_h = int(h_img * 0.30)
    red_mask[0:legend_h, 0:legend_w] = 0

    dot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dots_only = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, dot_kernel)

    contours_data = cv2.findContours(dots_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

    dot_centers: list[tuple[int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        x, y, box_w, box_h = cv2.boundingRect(contour)

        if area <= 2000:
            if circularity < 0.3:
                continue
        else:
            fill_ratio = area / float(max(1, box_w * box_h))
            if fill_ratio < 0.15:
                continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        estimated_dots = max(1, int(area / 300))
        for _ in range(estimated_dots):
            dot_centers.append((center_x, center_y))

    if not dot_centers:
        return []

    max_dist = max(w_img, h_img) * 0.08
    groups = group_exact_red_dot_centers(dot_centers, max_dist=max_dist)

    summaries: list[dict[str, object]] = []
    for group_dots in groups:
        if len(group_dots) < min_dot_count:
            continue

        xs = [point[0] for point in group_dots]
        ys = [point[1] for point in group_dots]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min
        if max(width, height) < 12:
            continue
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        radius = int(max(width, height) / 2) + padding
        summaries.append(
            {
                "indexes": list(range(len(group_dots))),
                "components": [],
                "total_area": int(width * height),
                "max_extent": float(max(width, height)),
                "sort_key": (float(x_min), float((y_min + y_max) / 2.0)),
                "circle": (float(center_x), float(center_y), float(radius)),
                "circles": [(float(center_x), float(center_y), float(radius))],
                "dot_count": len(group_dots),
                "route_score": float(len(group_dots)),
            }
        )

    return summaries


def group_exact_red_dot_centers(
    dot_centers: Sequence[tuple[int, int]],
    max_dist: float,
) -> list[list[tuple[int, int]]]:
    if not dot_centers:
        return []

    parent = list(range(len(dot_centers)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    max_dist_squared = max_dist * max_dist
    for left_index in range(len(dot_centers)):
        for right_index in range(left_index + 1, len(dot_centers)):
            dx = dot_centers[left_index][0] - dot_centers[right_index][0]
            dy = dot_centers[left_index][1] - dot_centers[right_index][1]
            if (dx * dx + dy * dy) <= max_dist_squared:
                union(left_index, right_index)

    groups: dict[int, list[tuple[int, int]]] = {}
    for index, center in enumerate(dot_centers):
        root = find(index)
        groups.setdefault(root, []).append(center)

    return sorted(groups.values(), key=len, reverse=True)


def is_kpi_measurement_component(component: dict[str, object] | DotComponent) -> bool:
    component = build_component_like(component)
    width = component.width
    height = component.height
    area = component.area
    if area < KPI_COMPONENT_MIN_PIXELS or area > KPI_COMPONENT_MAX_PIXELS:
        return False
    if width < KPI_COMPONENT_MIN_SPAN or height < KPI_COMPONENT_MIN_SPAN:
        return False
    if width > KPI_COMPONENT_MAX_SPAN or height > KPI_COMPONENT_MAX_SPAN:
        return False

    fill_ratio = component.fill_ratio
    if fill_ratio < KPI_COMPONENT_MIN_FILL:
        return False

    aspect_ratio = width / float(max(1, height))
    return KPI_COMPONENT_ASPECT_MIN <= aspect_ratio <= KPI_COMPONENT_ASPECT_MAX


def is_kpi_measurement_component_stats(component: dict[str, object] | DotComponent) -> bool:
    if isinstance(component, DotComponent):
        return is_kpi_measurement_component(component)

    area = int(component.get("area", 0))
    width = int(component.get("width", 0))
    height = int(component.get("height", 0))
    if area < KPI_COMPONENT_MIN_PIXELS or area > KPI_COMPONENT_MAX_PIXELS:
        return False
    if width < KPI_COMPONENT_MIN_SPAN or height < KPI_COMPONENT_MIN_SPAN:
        return False
    if width > KPI_COMPONENT_MAX_SPAN or height > KPI_COMPONENT_MAX_SPAN:
        return False

    fill_ratio = area / float(max(1, width * height))
    if fill_ratio < KPI_COMPONENT_MIN_FILL:
        return False

    aspect_ratio = width / float(max(1, height))
    return KPI_COMPONENT_ASPECT_MIN <= aspect_ratio <= KPI_COMPONENT_ASPECT_MAX


def build_component_like(component: dict[str, object] | DotComponent) -> DotComponent:
    if isinstance(component, DotComponent):
        return component

    pixels = component.get("pixels", [])
    bbox = component.get("bbox")
    area = int(component.get("area", len(pixels)))
    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        width = int(component.get("width", max_x - min_x + 1))
        height = int(component.get("height", max_y - min_y + 1))
    else:
        xs = [pixel[0] for pixel in pixels]
        ys = [pixel[1] for pixel in pixels]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
    fill_ratio = area / float(max(1, width * height))
    xs_array = component.get("xs")
    ys_array = component.get("ys")
    raw_center = component.get("center")
    if raw_center is not None:
        center = (float(raw_center[0]), float(raw_center[1]))
    elif xs_array is not None and ys_array is not None and len(xs_array) == area:
        center = (float(xs_array.mean()), float(ys_array.mean()))
    else:
        center = (
            sum(pixel[0] for pixel in pixels) / area,
            sum(pixel[1] for pixel in pixels) / area,
        )
    return DotComponent(
        pixels=pixels,
        area=area,
        bbox=(min_x, min_y, max_x, max_y),
        center=center,
        width=width,
        height=height,
        fill_ratio=fill_ratio,
        mean_rgb=(0.0, 0.0, 0.0),
        mean_lab=(0.0, 0.0, 0.0),
    )


def is_red_component(bitmap: Bitmap, component: dict[str, object] | DotComponent) -> bool:
    component = build_dot_component(bitmap, component) if not isinstance(component, DotComponent) or component.mean_rgb == (0.0, 0.0, 0.0) else component
    red, green, blue = component.mean_rgb
    hue, saturation, value = rgb_to_hsv(int(red), int(green), int(blue))

    if saturation < COLOR_SATURATION_THRESHOLD or value < COLOR_VALUE_THRESHOLD:
        return False

    hue_degrees = hue * 360.0
    red_dominant = red >= (green * 1.12) and red >= (blue * 1.35)
    return red_dominant and (hue_degrees <= RED_HUE_WINDOW_DEG or hue_degrees >= (360.0 - RED_HUE_WINDOW_DEG))


def extract_red_run_components(
    bitmap: Bitmap,
    point_components: Sequence[DotComponent] | None = None,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> list[DotComponent]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    legend_swatches = list(legend_swatches_override) if legend_swatches_override is not None else detect_legend_swatches(bitmap, legend_x, legend_y)
    degraded_swatch = degraded_swatch_override if degraded_swatch_override is not None else resolve_degraded_swatch(legend_swatches)
    candidates = list(point_components) if point_components is not None else extract_kpi_point_components(bitmap)

    min_fill_ratio = RED_RUN_COMPONENT_MIN_FILL if degraded_swatch_override is None else max(KPI_COMPONENT_MIN_FILL, RED_RUN_COMPONENT_MIN_FILL - 0.07)
    raw_components = []
    for component in candidates:
        red, green, blue = component.mean_rgb
        hue, saturation, value = rgb_to_hsv(int(red), int(green), int(blue))
        hue_degrees = hue * 360.0
        red_dominant = red >= (green * 1.08) and red >= (blue * 1.25)
        if not red_dominant:
            continue
        if saturation < 0.35 or value < 0.30:
            continue
        if not is_degraded_component_color(component, degraded_swatch, legend_swatches, hue_degrees):
            continue
        if (
            is_red_run_component(component, min_fill_ratio=min_fill_ratio)
            and not is_top_ui_noise_component(component, legend_x, legend_y)
        ):
            raw_components.append(component)
    if not raw_components:
        return []

    dominant_area = estimate_dominant_red_area(raw_components)
    min_area = max(RED_RUN_COMPONENT_MIN_PIXELS, dominant_area * RED_RUN_DOMINANT_AREA_RATIO_MIN)
    max_area = max(min_area, dominant_area * RED_RUN_DOMINANT_AREA_RATIO_MAX)

    filtered = [component for component in raw_components if min_area <= component.area <= max_area]
    return filtered or raw_components


def extract_visual_red_dot_components(
    bitmap: Bitmap,
    point_components: Sequence[DotComponent] | None = None,
) -> list[DotComponent]:
    candidates = list(point_components) if point_components is not None else extract_kpi_point_components(bitmap)
    indexes = extract_visual_red_dot_indexes(bitmap, candidates)
    return [candidates[index] for index in indexes]


def extract_visual_red_dot_indexes(
    bitmap: Bitmap,
    point_components: Sequence[DotComponent] | None = None,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> list[int]:
    raw_components = extract_red_run_components(
        bitmap,
        point_components,
        legend_swatches_override=legend_swatches_override,
        degraded_swatch_override=degraded_swatch_override,
    )
    if not raw_components:
        return []

    dominant_area = estimate_dominant_red_area(raw_components)
    min_area = max(RED_RUN_COMPONENT_MIN_PIXELS, dominant_area * 0.72)
    max_area = max(min_area, dominant_area * 1.28)
    filtered = [component for component in raw_components if min_area <= component.area <= max_area]
    selected = filtered or raw_components
    if point_components is None:
        return list(range(len(selected)))

    component_indexes = {id(component): index for index, component in enumerate(point_components)}
    return [component_indexes[id(component)] for component in selected if id(component) in component_indexes]


def detect_legend_degraded_hue(bitmap: Bitmap, legend_x: int, legend_y: int) -> float | None:
    swatches = detect_legend_swatches(bitmap, legend_x, legend_y)
    if not swatches:
        return None
    degraded_swatch = resolve_degraded_swatch(swatches)
    if degraded_swatch is None:
        return None
    return degraded_swatch.hue_degrees


def bitmap_has_degraded_legend_swatch(bitmap: Bitmap) -> bool:
    legend_x = int(bitmap.width * LEGEND_X_RATIO)
    legend_y = int(bitmap.height * LEGEND_Y_RATIO)
    swatches = detect_legend_swatches(bitmap, legend_x, legend_y)
    return resolve_degraded_swatch(swatches) is not None


def extract_bitmap_legend_reference(bitmap: Bitmap) -> tuple[list[LegendSwatch], LegendSwatch | None]:
    legend_x = int(bitmap.width * LEGEND_X_RATIO)
    legend_y = int(bitmap.height * LEGEND_Y_RATIO)
    swatches = detect_legend_swatches(bitmap, legend_x, legend_y)
    return swatches, resolve_degraded_swatch(swatches)


def detect_legend_swatches(bitmap: Bitmap, legend_x: int, legend_y: int) -> list[LegendSwatch]:
    swatch_x_limit = min(legend_x, max(24, int(bitmap.width * 0.08)))
    hsv_arrays = bitmap_hsv_arrays(bitmap)
    if hsv_arrays is not None:
        _hue, saturation, value = hsv_arrays
        roi_mask_array = (
            (saturation[:legend_y, :swatch_x_limit] >= COLOR_SATURATION_THRESHOLD)
            & (value[:legend_y, :swatch_x_limit] >= COLOR_VALUE_THRESHOLD)
        ).astype(np.uint8)
        roi_mask_rows = None
    else:
        roi_mask_rows = [[0] * swatch_x_limit for _ in range(legend_y)]
        for y in range(legend_y):
            for x in range(swatch_x_limit):
                red, green, blue = rgb_pixel(bitmap, x, y)
                hue, saturation, value = rgb_to_hsv(red, green, blue)
                if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                    roi_mask_rows[y][x] = 1
        roi_mask_array = None

    swatches: list[LegendSwatch] = []
    swatch_component_count = 0
    for raw_component in extract_binary_components(mask_rows=roi_mask_rows, mask_array=roi_mask_array):
        component = build_dot_component(bitmap, raw_component)
        min_x, min_y, max_x, max_y = component.bbox
        area = component.area
        width = component.width
        height = component.height
        if area < 20 or area > 200:
            continue
        if width < 4 or width > 16 or height < 4 or height > 16:
            continue
        swatch_component_count += 1

        red, green, blue = component.mean_rgb
        lab = component.mean_lab
        hue, saturation, value = rgb_to_hsv(int(red), int(green), int(blue))
        hue_degrees = hue * 360.0
        if saturation < 0.5 or value < 0.5:
            continue
        center_y = (min_y + max_y) / 2.0
        swatches.append(
            LegendSwatch(
                bbox=component.bbox,
                center_y=center_y,
                rgb=(red, green, blue),
                lab=lab,
                hue_degrees=hue_degrees,
                saturation=saturation,
                value=value,
            )
        )

    if swatch_component_count < 3 or not swatches:
        return []

    swatches.sort(key=lambda item: item.center_y)
    return swatches


def resolve_degraded_swatch(swatches: Sequence[LegendSwatch]) -> LegendSwatch | None:
    if not swatches:
        return None
    return max(swatches, key=lambda swatch: swatch.center_y)


def is_degraded_component_color(
    component: DotComponent,
    degraded_swatch: LegendSwatch | None,
    swatches: Sequence[LegendSwatch],
    hue_degrees: float,
) -> bool:
    if degraded_swatch is None or not swatches:
        return matches_degraded_hue(hue_degrees, None)

    distances = [
        (
            color_distance_lab(component.mean_lab, swatch.lab),
            swatch,
        )
        for swatch in swatches
    ]
    distances.sort(key=lambda item: item[0])
    nearest_distance, nearest_swatch = distances[0]
    if color_distance_lab(nearest_swatch.lab, degraded_swatch.lab) > 1.0:
        return False
    non_degraded_distances = [
        distance
        for distance, swatch in distances[1:]
        if color_distance_lab(swatch.lab, degraded_swatch.lab) > 1.0
    ]
    if not non_degraded_distances:
        return True

    second_distance = non_degraded_distances[0]
    return nearest_distance <= second_distance * LEGEND_DISTANCE_MARGIN_RATIO


def matches_degraded_hue(hue_degrees: float, legend_degraded_hue: float | None) -> bool:
    if legend_degraded_hue is None:
        return hue_degrees <= 40.0 or hue_degrees >= 345.0
    return circular_hue_distance(hue_degrees, legend_degraded_hue) <= LEGEND_HUE_WINDOW_DEG


def circular_hue_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def color_distance_lab(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def rgb_to_lab(red: float, green: float, blue: float) -> tuple[float, float, float]:
    def to_linear(channel: float) -> float:
        channel = channel / 255.0
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red_linear = to_linear(red)
    green_linear = to_linear(green)
    blue_linear = to_linear(blue)

    x = (0.4124564 * red_linear) + (0.3575761 * green_linear) + (0.1804375 * blue_linear)
    y = (0.2126729 * red_linear) + (0.7151522 * green_linear) + (0.0721750 * blue_linear)
    z = (0.0193339 * red_linear) + (0.1191920 * green_linear) + (0.9503041 * blue_linear)

    x /= 0.95047
    z /= 1.08883

    def f(value: float) -> float:
        if value > 0.008856:
            return value ** (1.0 / 3.0)
        return (7.787 * value) + (16.0 / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return (l, a, b)


def is_top_ui_noise_component(
    component: dict[str, object] | DotComponent,
    legend_x: int,
    legend_y: int,
) -> bool:
    min_x, min_y, max_x, max_y = component_bbox(component)
    _ = min_y
    # Only treat components as top-UI noise when they live in the upper-left
    # strip around the legend/header. Real route dots can appear above the
    # legend baseline on some VoLTE maps, but they are well to the right.
    return max_y < legend_y and min_x < int(legend_x * 1.15) and max_x < int(legend_x * 1.35)


def is_red_run_component(
    component: dict[str, object] | DotComponent,
    min_fill_ratio: float = RED_RUN_COMPONENT_MIN_FILL,
) -> bool:
    component = build_component_like(component)
    width = component.width
    height = component.height
    area = component.area
    if area < RED_RUN_COMPONENT_MIN_PIXELS or area > RED_RUN_COMPONENT_MAX_PIXELS:
        return False
    if width < RED_RUN_COMPONENT_MIN_SPAN or height < RED_RUN_COMPONENT_MIN_SPAN:
        return False
    if width > RED_RUN_COMPONENT_MAX_SPAN or height > RED_RUN_COMPONENT_MAX_SPAN:
        return False

    fill_ratio = component.fill_ratio
    if fill_ratio < min_fill_ratio:
        return False

    aspect_ratio = width / float(max(1, height))
    return RED_RUN_COMPONENT_ASPECT_MIN <= aspect_ratio <= RED_RUN_COMPONENT_ASPECT_MAX


def component_has_bright_inner_symbol(
    bitmap: Bitmap,
    component: dict[str, object] | DotComponent,
) -> bool:
    normalized = build_component_like(component)
    width = normalized.width
    height = normalized.height
    if width < 12 or height < 12:
        return False

    pixel_set = set(normalized.pixels)
    center_x, center_y = normalized.center
    radius = max(2.5, min(width, height) * 0.18)
    inner_white_pixels = 0
    sampled_pixels = 0
    min_x, min_y, max_x, max_y = normalized.bbox
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if math.hypot(x - center_x, y - center_y) > radius:
                continue
            sampled_pixels += 1
            if (x, y) in pixel_set:
                continue
            red, green, blue = rgb_pixel(bitmap, x, y)
            if red >= 225 and green >= 225 and blue >= 225:
                inner_white_pixels += 1

    if sampled_pixels <= 0:
        return False
    return inner_white_pixels >= INNER_WHITE_SYMBOL_MIN_PIXELS


def is_solid_circle_like_component(
    bitmap: Bitmap,
    component: dict[str, object] | DotComponent,
) -> bool:
    normalized = build_component_like(component)
    width = normalized.width
    height = normalized.height
    if width < 6 or height < 6:
        return False

    aspect_ratio = width / float(max(1, height))
    if aspect_ratio < OVERLAY_POINT_ASPECT_MIN or aspect_ratio > OVERLAY_POINT_ASPECT_MAX:
        return False

    fill_ratio = normalized.fill_ratio
    if fill_ratio < OVERLAY_POINT_FILL_MIN or fill_ratio > OVERLAY_POINT_FILL_MAX:
        return False

    pixel_set = set(normalized.pixels)
    center_x, center_y = normalized.center
    inner_radius = max(2.5, min(width, height) * 0.24)
    sampled_pixels = 0
    solid_pixels = 0
    white_gap_pixels = 0
    min_x, min_y, max_x, max_y = normalized.bbox
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if math.hypot(x - center_x, y - center_y) > inner_radius:
                continue
            sampled_pixels += 1
            if (x, y) in pixel_set:
                solid_pixels += 1
                continue
            red, green, blue = rgb_pixel(bitmap, x, y)
            if red >= 220 and green >= 220 and blue >= 220:
                white_gap_pixels += 1

    if sampled_pixels <= 0:
        return False

    solid_ratio = solid_pixels / float(sampled_pixels)
    white_gap_ratio = white_gap_pixels / float(sampled_pixels)
    if solid_ratio < OVERLAY_POINT_CENTER_SOLID_MIN_RATIO:
        return False
    if white_gap_ratio > OVERLAY_POINT_CENTER_WHITE_MAX_RATIO:
        return False
    return True


def build_direct_mask_route_run_summaries(
    bitmap: Bitmap,
    legend_swatches_override: Sequence[LegendSwatch] | None = None,
    degraded_swatch_override: LegendSwatch | None = None,
) -> list[dict[str, object]]:
    components = extract_direct_degraded_components(
        bitmap,
        legend_swatches_override=legend_swatches_override,
        degraded_swatch_override=degraded_swatch_override,
    )
    if len(components) < MIN_CONTINUOUS_RED_POINTS:
        return []

    link_distance = estimate_direct_overlay_cluster_distance(components)
    clusters = cluster_components(components, link_distance=link_distance)
    summaries: list[dict[str, object]] = []
    for cluster in clusters:
        route_score = coverage_overlay_cluster_score(cluster, bitmap)
        if route_score <= 0.0:
            continue
        summary = build_run_summary(cluster, segmented=True, bitmap=bitmap)
        annotation_cluster = select_route_annotation_cluster(cluster, bitmap)
        annotation_circles = route_annotation_circles(annotation_cluster) if annotation_cluster else []
        annotation_circle = route_annotation_circle(annotation_cluster) if annotation_cluster else None
        if annotation_circles:
            summary["circle"] = annotation_circle
            summary["circles"] = annotation_circles
        elif annotation_circle is not None:
            summary["circle"] = annotation_circle
            summary["circles"] = [annotation_circle]
        summary["route_score"] = route_score
        summaries.append(summary)
    return summaries


def estimate_direct_overlay_cluster_distance(
    components: Sequence[dict[str, object] | DotComponent],
) -> float:
    base_distance = estimate_degraded_component_cluster_distance(components)
    adaptive_distance = base_distance * DIRECT_OVERLAY_CLUSTER_MULTIPLIER
    return max(DIRECT_OVERLAY_CLUSTER_MIN, min(DIRECT_OVERLAY_CLUSTER_MAX, adaptive_distance))


def coverage_overlay_cluster_score(
    cluster: Sequence[dict[str, object] | DotComponent],
    bitmap: Bitmap,
) -> float:
    if len(cluster) < MIN_CONTINUOUS_RED_POINTS:
        return -1.0
    if is_text_like_label_cluster(list(cluster)):
        return -1.0

    components = [build_component_like(component) for component in cluster]
    areas = [component.area for component in components]
    area_mean = statistics.mean(areas)
    if area_mean <= 0:
        return -1.0
    area_cv = statistics.pstdev(areas) / area_mean if len(areas) > 1 else 0.0
    if area_cv > 2.0:
        return -1.0

    centers = [component.center for component in components]
    distances = nearest_neighbor_distances_for_centers(centers)
    if not distances:
        return -1.0
    distance_mean = statistics.mean(distances)
    if distance_mean <= 0:
        return -1.0
    distance_cv = statistics.pstdev(distances) / distance_mean if len(distances) > 1 else 0.0
    if distance_cv > 1.1:
        return -1.0

    symbol_count = sum(component_has_bright_inner_symbol(bitmap, component) for component in components)
    symbol_ratio = symbol_count / float(len(components))
    if symbol_ratio > 0.2:
        return -1.0

    x_values = [center[0] for center in centers]
    y_values = [center[1] for center in centers]
    width_extent = max(x_values) - min(x_values)
    height_extent = max(y_values) - min(y_values)
    if width_extent < 120.0 or height_extent < 80.0:
        return -1.0
    mean_y = statistics.mean(y_values)
    if mean_y < (bitmap.height * 0.45):
        return -1.0

    hue_distances: list[float] = []
    for component in components:
        hue, _saturation, _value = rgb_to_hsv(int(component.mean_rgb[0]), int(component.mean_rgb[1]), int(component.mean_rgb[2]))
        hue_distances.append(circular_hue_distance(hue * 360.0, 0.0))
    mean_red_distance = statistics.mean(hue_distances)
    if mean_red_distance > 30.0:
        return -1.0

    extent_bonus = min(width_extent, height_extent) / 25.0
    return (
        (len(components) * 8.0)
        + extent_bonus
        - (mean_red_distance * 1.8)
        - (area_cv * 2.0)
        - (distance_cv * 2.0)
        - (symbol_ratio * 10.0)
    )


def select_route_annotation_cluster(
    cluster: Sequence[dict[str, object] | DotComponent],
    bitmap: Bitmap,
) -> list[DotComponent]:
    components = [build_component_like(component) for component in cluster]
    filtered_components = [
        component
        for component in components
        if not component_has_bright_inner_symbol(bitmap, component)
    ]
    candidates = filtered_components or components
    if len(candidates) < MIN_CONTINUOUS_RED_POINTS:
        return candidates

    annotation_link_distance = max(
        24.0,
        min(84.0, estimate_degraded_component_cluster_distance(candidates) * 1.08),
    )
    subclusters = cluster_components(candidates, link_distance=annotation_link_distance)
    if len(subclusters) <= 1:
        return candidates

    scored_clusters: list[tuple[float, int, float, list[DotComponent]]] = []
    for subcluster in subclusters:
        route_score = coverage_overlay_cluster_score(subcluster, bitmap)
        if route_score <= 0.0:
            continue
        scored_clusters.append(
            (
                route_score,
                len(subcluster),
                cluster_max_extent(list(subcluster)),
                [build_component_like(component) for component in subcluster],
            )
        )

    if not scored_clusters:
        return candidates

    scored_clusters.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    best_cluster = scored_clusters[0][3]
    loop_core = extract_route_loop_core(best_cluster)
    if len(loop_core) >= MIN_CONTINUOUS_RED_POINTS:
        return loop_core
    return best_cluster


def extract_route_loop_core(
    components: Sequence[DotComponent],
) -> list[DotComponent]:
    normalized = [build_component_like(component) for component in components]
    if len(normalized) < MIN_CONTINUOUS_RED_POINTS + 2:
        return normalized

    link_distance = max(
        30.0,
        min(96.0, estimate_degraded_component_cluster_distance(normalized) * 1.2),
    )
    centers = [component.center for component in normalized]
    distance_matrix = pairwise_distance_matrix(centers)
    adjacency: list[set[int]] = [set() for _ in normalized]
    for left_index in range(len(normalized)):
        for right_index in range(left_index + 1, len(normalized)):
            if distance_matrix[left_index][right_index] > link_distance:
                continue
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)

    active = set(range(len(normalized)))
    changed = True
    while changed and len(active) > MIN_CONTINUOUS_RED_POINTS:
        changed = False
        leaves = [index for index in active if len(adjacency[index] & active) <= 1]
        if not leaves:
            break
        removable = len(active) - MIN_CONTINUOUS_RED_POINTS
        if removable <= 0:
            break
        for index in leaves[:removable]:
            if index in active:
                active.remove(index)
                changed = True

    if len(active) < MIN_CONTINUOUS_RED_POINTS:
        return normalized

    remaining_indexes = sorted(active, key=lambda index: (normalized[index].center[0], normalized[index].center[1]))
    return [normalized[index] for index in remaining_indexes]


def route_annotation_circle(
    components: Sequence[DotComponent],
) -> tuple[float, float, float] | None:
    if not components:
        return None

    normalized = [build_component_like(component) for component in components]
    normalized = select_dense_annotation_core(normalized)
    center_x = sum(component.center[0] for component in normalized) / len(normalized)
    center_y = sum(component.center[1] for component in normalized) / len(normalized)
    base_radius = max(
        math.hypot(component.center[0] - center_x, component.center[1] - center_y)
        + (max(component.width, component.height) * 0.6)
        for component in normalized
    )
    radius = max(18.0, base_radius + 14.0)
    return center_x, center_y, radius


def route_annotation_circles(
    components: Sequence[DotComponent],
) -> list[tuple[float, float, float]]:
    normalized = [build_component_like(component) for component in components]
    if not normalized:
        return []
    if len(normalized) < MIN_CONTINUOUS_RED_POINTS:
        circle = route_annotation_circle(normalized)
        return [] if circle is None else [circle]

    return component_hotspot_circles(normalized)


def select_dense_annotation_core(
    components: Sequence[DotComponent],
) -> list[DotComponent]:
    normalized = [build_component_like(component) for component in components]
    if len(normalized) < (MIN_CONTINUOUS_RED_POINTS + 3):
        return normalized

    centers = [component.center for component in normalized]
    distance_matrix = pairwise_distance_matrix(centers)
    local_scores: list[tuple[float, int]] = []
    for index in range(len(normalized)):
        neighbor_distances = sorted(
            distance
            for other_index, distance in enumerate(distance_matrix[index])
            if other_index != index
        )
        if not neighbor_distances:
            continue
        sample = neighbor_distances[: min(3, len(neighbor_distances))]
        local_scores.append((sum(sample) / len(sample), index))

    if len(local_scores) < len(normalized):
        return normalized

    score_values = [score for score, _index in local_scores]
    score_median = statistics.median(score_values)
    trim_candidates = [
        index
        for score, index in sorted(local_scores, reverse=True)
        if score > (score_median * 1.22)
    ]
    max_trim = min(3, max(1, len(normalized) // 6))
    trim_indexes = set(trim_candidates[:max_trim])
    if not trim_indexes:
        return normalized

    trimmed = [component for index, component in enumerate(normalized) if index not in trim_indexes]
    if len(trimmed) < MIN_CONTINUOUS_RED_POINTS:
        return normalized
    return trimmed


def estimate_dominant_red_area(components: Sequence[dict[str, object] | DotComponent]) -> float:
    return estimate_dominant_component_area(components)


def estimate_dominant_component_area(components: Sequence[dict[str, object] | DotComponent]) -> float:
    if not components:
        return float(RED_RUN_COMPONENT_MIN_PIXELS)

    areas = [build_component_like(component).area for component in components]
    if not areas:
        return float(RED_RUN_COMPONENT_MIN_PIXELS)
    largest_areas = sorted(areas, reverse=True)[:10]
    if np is not None:
        return float(np.median(np.asarray(largest_areas, dtype=np.float32)))
    return float(statistics.median(largest_areas))


def estimate_red_run_link_distance(components: Sequence[dict[str, object] | DotComponent]) -> float:
    if len(components) < 2:
        return ADAPTIVE_LINK_DISTANCE_MIN

    centers = [component_center(component) for component in components]
    nearest_neighbor_distances: list[float] = []
    for index, (x1, y1) in enumerate(centers):
        distances = [
            math.hypot(x2 - x1, y2 - y1)
            for other_index, (x2, y2) in enumerate(centers)
            if other_index != index
        ]
        if not distances:
            continue
        nearest_neighbor_distances.append(min(distances))

    if not nearest_neighbor_distances:
        return ADAPTIVE_LINK_DISTANCE_MIN

    adaptive_distance = statistics.median(nearest_neighbor_distances) * ADAPTIVE_LINK_DISTANCE_MULTIPLIER
    return max(ADAPTIVE_LINK_DISTANCE_MIN, min(ADAPTIVE_LINK_DISTANCE_MAX, adaptive_distance))


def estimate_degraded_component_cluster_distance(
    components: Sequence[dict[str, object] | DotComponent],
) -> float:
    if len(components) < 2:
        return DEGRADED_COMPONENT_CLUSTER_MIN

    centers = [component_center(component) for component in components]
    nearest_neighbor_distances = nearest_neighbor_distances_for_centers(centers)
    if not nearest_neighbor_distances:
        return DEGRADED_COMPONENT_CLUSTER_MIN

    adaptive_distance = statistics.median(nearest_neighbor_distances) * DEGRADED_COMPONENT_CLUSTER_MULTIPLIER
    return max(DEGRADED_COMPONENT_CLUSTER_MIN, min(DEGRADED_COMPONENT_CLUSTER_MAX, adaptive_distance))


def estimate_dot_chain_link_distance(components: Sequence[dict[str, object] | DotComponent]) -> float:
    if len(components) < 2:
        return DOT_CHAIN_LINK_DISTANCE_MIN

    centers = [component_center(component) for component in components]
    nearest_neighbor_distances = nearest_neighbor_distances_for_centers(centers)
    if not nearest_neighbor_distances:
        return DOT_CHAIN_LINK_DISTANCE_MIN

    adaptive_distance = statistics.median(nearest_neighbor_distances) * DOT_CHAIN_LINK_DISTANCE_MULTIPLIER
    return max(DOT_CHAIN_LINK_DISTANCE_MIN, min(DOT_CHAIN_LINK_DISTANCE_MAX, adaptive_distance))


def build_ordered_dot_chains(
    components: Sequence[dict[str, object] | DotComponent],
    link_distance: float,
) -> list[list[DotComponent]]:
    normalized = [build_component_like(component) for component in components]
    chain_indexes = build_ordered_dot_chain_indexes(normalized, link_distance)
    return [[normalized[index] for index in chain] for chain in chain_indexes]


def build_ordered_dot_chain_indexes(
    components: Sequence[dict[str, object] | DotComponent],
    link_distance: float,
) -> list[list[int]]:
    normalized = [build_component_like(component) for component in components]
    if not normalized:
        return []

    centers = [component.center for component in normalized]
    distance_matrix = pairwise_distance_matrix(centers)
    neighbor_lists: list[list[int]] = []
    for index in range(len(normalized)):
        candidates = [
            other_index
            for other_index, distance in enumerate(distance_matrix[index])
            if other_index != index and distance <= link_distance
        ]
        candidates.sort(key=lambda other_index: distance_matrix[index][other_index])
        neighbor_lists.append(candidates)

    reciprocal_neighbors = build_reciprocal_neighbor_sets(neighbor_lists)
    adjacency = prune_chain_adjacency(reciprocal_neighbors, distance_matrix, centers)

    visited = [False] * len(normalized)
    chains: list[list[int]] = []
    for index in range(len(normalized)):
        if visited[index]:
            continue
        stack = [index]
        component_indexes: list[int] = []
        visited[index] = True
        while stack:
            current = stack.pop()
            component_indexes.append(current)
            for neighbor in adjacency[current]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        chains.append(order_dot_chain_indexes(component_indexes, adjacency, normalized))

    chains = merge_chain_endpoint_indexes(chains, normalized, link_distance)
    chains.sort(key=lambda chain: (-len(chain), chain_sort_key_from_indexes(chain, normalized)[0], chain_sort_key_from_indexes(chain, normalized)[1]))
    return chains


def build_reciprocal_neighbor_sets(neighbor_lists: Sequence[Sequence[int]]) -> list[set[int]]:
    reciprocal: list[set[int]] = [set() for _ in neighbor_lists]
    for index, neighbors in enumerate(neighbor_lists):
        rank_lookup = {neighbor: rank for rank, neighbor in enumerate(neighbors[:DOT_CHAIN_NEIGHBOR_RANK_LIMIT])}
        for neighbor, rank in rank_lookup.items():
            if index not in neighbor_lists[neighbor][:DOT_CHAIN_NEIGHBOR_RANK_LIMIT]:
                continue
            reciprocal[index].add(neighbor)
            reciprocal[neighbor].add(index)
    return reciprocal


def prune_chain_adjacency(
    reciprocal_neighbors: Sequence[set[int]],
    distance_matrix: Sequence[Sequence[float]],
    centers: Sequence[tuple[float, float]],
) -> list[set[int]]:
    selected_neighbors: list[set[int]] = [set() for _ in reciprocal_neighbors]
    for index, neighbors in enumerate(reciprocal_neighbors):
        if not neighbors:
            continue
        neighbor_list = sorted(neighbors, key=lambda other_index: distance_matrix[index][other_index])
        best_score = -math.inf
        best_subset: tuple[int, ...] = ()
        candidate_subsets: list[tuple[int, ...]] = [()]
        candidate_subsets.extend((neighbor,) for neighbor in neighbor_list)
        for left_pos in range(len(neighbor_list)):
            for right_pos in range(left_pos + 1, len(neighbor_list)):
                candidate_subsets.append((neighbor_list[left_pos], neighbor_list[right_pos]))

        for subset in candidate_subsets:
            score = score_neighbor_subset(index, subset, distance_matrix, centers)
            if score > best_score:
                best_score = score
                best_subset = subset
        selected_neighbors[index] = set(best_subset)

    adjacency: list[set[int]] = [set() for _ in reciprocal_neighbors]
    for index, neighbors in enumerate(selected_neighbors):
        for neighbor in neighbors:
            if index not in selected_neighbors[neighbor]:
                continue
            adjacency[index].add(neighbor)
            adjacency[neighbor].add(index)
    return adjacency


def score_neighbor_subset(
    index: int,
    subset: Sequence[int],
    distance_matrix: Sequence[Sequence[float]],
    centers: Sequence[tuple[float, float]],
) -> float:
    if not subset:
        return 0.0

    distance_score = sum(1.0 / max(1.0, distance_matrix[index][neighbor]) for neighbor in subset)
    if len(subset) == 1:
        return distance_score

    left_neighbor, right_neighbor = subset
    angle_degrees = abs(
        vector_angle_degrees(
            vector_between(centers[index], centers[left_neighbor]),
            vector_between(centers[index], centers[right_neighbor]),
        )
    )
    if angle_degrees < DOT_CHAIN_MIN_PAIR_ANGLE_DEG:
        return -math.inf
    angle_bonus = (angle_degrees - DOT_CHAIN_MIN_PAIR_ANGLE_DEG) / max(1.0, 180.0 - DOT_CHAIN_MIN_PAIR_ANGLE_DEG)
    return distance_score + angle_bonus


def order_dot_chain(
    component_indexes: Sequence[int],
    adjacency: Sequence[set[int]],
    components: Sequence[DotComponent],
) -> list[DotComponent]:
    ordered_indexes = order_dot_chain_indexes(component_indexes, adjacency, components)
    return [components[index] for index in ordered_indexes]


def order_dot_chain_indexes(
    component_indexes: Sequence[int],
    adjacency: Sequence[set[int]],
    components: Sequence[DotComponent],
) -> list[int]:
    local_indexes = set(component_indexes)
    endpoint_candidates = [index for index in component_indexes if len(adjacency[index] & local_indexes) <= 1]
    if endpoint_candidates:
        start_index = min(endpoint_candidates, key=lambda index: (components[index].center[0], components[index].center[1]))
    else:
        start_index = min(component_indexes, key=lambda index: (components[index].center[0], components[index].center[1]))

    ordered_indexes: list[int] = []
    visited: set[int] = set()
    previous_index: int | None = None
    current_index: int | None = start_index

    while current_index is not None and current_index not in visited:
        ordered_indexes.append(current_index)
        visited.add(current_index)
        next_candidates = [index for index in adjacency[current_index] if index in local_indexes and index not in visited]
        if not next_candidates:
            current_index = None
            continue
        if previous_index is None or len(next_candidates) == 1:
            next_index = min(
                next_candidates,
                key=lambda index: math.hypot(
                    components[index].center[0] - components[current_index].center[0],
                    components[index].center[1] - components[current_index].center[1],
                ),
            )
        else:
            previous_vector = (
                components[current_index].center[0] - components[previous_index].center[0],
                components[current_index].center[1] - components[previous_index].center[1],
            )
            next_index = max(
                next_candidates,
                key=lambda index: chain_direction_score(previous_vector, components[current_index].center, components[index].center),
            )
        previous_index, current_index = current_index, next_index

    if len(ordered_indexes) != len(component_indexes):
        remaining_indexes = sorted(
            (index for index in component_indexes if index not in visited),
            key=lambda index: (components[index].center[0], components[index].center[1]),
        )
        ordered_indexes.extend(remaining_indexes)

    return ordered_indexes


def chain_direction_score(
    previous_vector: tuple[float, float],
    current_center: tuple[float, float],
    next_center: tuple[float, float],
) -> float:
    next_vector = (next_center[0] - current_center[0], next_center[1] - current_center[1])
    previous_length = math.hypot(*previous_vector)
    next_length = math.hypot(*next_vector)
    if previous_length == 0 or next_length == 0:
        return 0.0
    dot = (previous_vector[0] * next_vector[0]) + (previous_vector[1] * next_vector[1])
    return dot / (previous_length * next_length)


def merge_chain_endpoints(
    chains: Sequence[Sequence[DotComponent]],
    link_distance: float,
) -> list[list[DotComponent]]:
    if not chains:
        return []
    flattened: list[DotComponent] = []
    indexed_chains: list[list[int]] = []
    offset = 0
    for chain in chains:
        flattened.extend(chain)
        indexed_chains.append(list(range(offset, offset + len(chain))))
        offset += len(chain)
    merged_indexes = merge_chain_endpoint_indexes(
        indexed_chains,
        flattened,
        link_distance,
    )
    return [[flattened[index] for index in chain] for chain in merged_indexes]


def merge_chain_endpoint_indexes(
    chains: Sequence[Sequence[int]],
    components: Sequence[DotComponent],
    link_distance: float,
) -> list[list[int]]:
    merged = [list(chain) for chain in chains if chain]
    if len(merged) < 2:
        return merged

    merge_distance = min(DOT_CHAIN_MERGE_DISTANCE_MAX, max(link_distance * DOT_CHAIN_MERGE_DISTANCE_MULTIPLIER, link_distance + 32.0))

    while True:
        best_candidate: tuple[float, int, int, list[int]] | None = None
        for left_index in range(len(merged)):
            for right_index in range(left_index + 1, len(merged)):
                candidate = evaluate_chain_merge_indexes(merged[left_index], merged[right_index], components, merge_distance)
                if candidate is None:
                    continue
                score, combined = candidate
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, left_index, right_index, combined)

        if best_candidate is None:
            break

        _score, left_index, right_index, combined = best_candidate
        merged[left_index] = combined
        merged.pop(right_index)

    return merged


def evaluate_chain_merge(
    left_chain: Sequence[DotComponent],
    right_chain: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[DotComponent]] | None:
    if not left_chain or not right_chain:
        return None
    combined_components = [component for component in left_chain] + [component for component in right_chain]
    left_indexes = list(range(len(left_chain)))
    right_indexes = list(range(len(left_chain), len(combined_components)))
    result = evaluate_chain_merge_indexes(left_indexes, right_indexes, combined_components, merge_distance)
    if result is None:
        return None
    score, combined = result
    return score, [combined_components[index] for index in combined]


def evaluate_chain_merge_indexes(
    left_chain: Sequence[int],
    right_chain: Sequence[int],
    components: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[int]] | None:
    best: tuple[float, list[int]] | None = None
    orientations = [
        (list(left_chain), list(right_chain)),
        (list(left_chain), list(reversed(right_chain))),
        (list(reversed(left_chain)), list(right_chain)),
        (list(reversed(left_chain)), list(reversed(right_chain))),
    ]
    for oriented_left, oriented_right in orientations:
        candidate = score_chain_merge_orientation_indexes(oriented_left, oriented_right, components, merge_distance)
        if candidate is None:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def score_chain_merge_orientation(
    left_chain: Sequence[DotComponent],
    right_chain: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[DotComponent]] | None:
    if not left_chain or not right_chain:
        return None
    combined_components = [component for component in left_chain] + [component for component in right_chain]
    left_indexes = list(range(len(left_chain)))
    right_indexes = list(range(len(left_chain), len(combined_components)))
    result = score_chain_merge_orientation_indexes(left_indexes, right_indexes, combined_components, merge_distance)
    if result is None:
        return None
    score, combined = result
    return score, [combined_components[index] for index in combined]


def score_chain_merge_orientation_indexes(
    left_chain: Sequence[int],
    right_chain: Sequence[int],
    components: Sequence[DotComponent],
    merge_distance: float,
) -> tuple[float, list[int]] | None:
    if not left_chain or not right_chain:
        return None

    left_end = components[left_chain[-1]].center
    right_start = components[right_chain[0]].center
    bridge_vector = vector_between(left_end, right_start)
    bridge_distance = math.hypot(*bridge_vector)
    if bridge_distance == 0.0 or bridge_distance > merge_distance:
        return None

    left_outward = chain_endpoint_outward_vector_indexes(left_chain, components, at_start=False)
    right_outward = chain_endpoint_outward_vector_indexes(right_chain, components, at_start=True)
    left_alignment = vector_cosine(left_outward, bridge_vector) if left_outward is not None else 1.0
    right_alignment = vector_cosine(right_outward, (-bridge_vector[0], -bridge_vector[1])) if right_outward is not None else 1.0
    if left_alignment < DOT_CHAIN_MERGE_ALIGNMENT_MIN or right_alignment < DOT_CHAIN_MERGE_ALIGNMENT_MIN:
        return None

    bridge_alignment = min(left_alignment, right_alignment)
    if bridge_alignment < DOT_CHAIN_MERGE_ALIGNMENT_GOOD and bridge_distance > (merge_distance * 0.7):
        return None

    merged_chain = list(left_chain) + list(right_chain)
    score = (
        bridge_alignment * 10.0
        + min(left_alignment, 1.0)
        + min(right_alignment, 1.0)
        + (len(merged_chain) / 100.0)
        - (bridge_distance / max(1.0, merge_distance))
    )
    return score, merged_chain


def chain_endpoint_outward_vector(
    chain: Sequence[DotComponent],
    at_start: bool,
) -> tuple[float, float] | None:
    if not chain:
        return None
    return chain_endpoint_outward_vector_indexes(list(range(len(chain))), chain, at_start)


def chain_endpoint_outward_vector_indexes(
    chain: Sequence[int],
    components: Sequence[DotComponent],
    at_start: bool,
) -> tuple[float, float] | None:
    if len(chain) < 2:
        return None
    sample_count = min(3, len(chain) - 1)
    if at_start:
        anchor = components[chain[0]].center
        neighbor = components[chain[sample_count]].center
        return (anchor[0] - neighbor[0], anchor[1] - neighbor[1])
    anchor = components[chain[-1]].center
    neighbor = components[chain[-1 - sample_count]].center
    return (anchor[0] - neighbor[0], anchor[1] - neighbor[1])


def vector_between(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    return (end[0] - start[0], end[1] - start[1])


def vector_cosine(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    left_length = math.hypot(*left)
    right_length = math.hypot(*right)
    if left_length == 0.0 or right_length == 0.0:
        return 0.0
    return ((left[0] * right[0]) + (left[1] * right[1])) / (left_length * right_length)


def vector_angle_degrees(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    cosine = max(-1.0, min(1.0, vector_cosine(left, right)))
    return math.degrees(math.acos(cosine))


def extract_qualifying_degraded_runs(
    dot_chains: Sequence[Sequence[DotComponent]],
    degraded_component_ids: set[int],
    link_distance: float,
) -> list[list[DotComponent]]:
    flat_components: list[DotComponent] = []
    chains_as_indexes: list[list[int]] = []
    offset = 0
    for chain in dot_chains:
        flat_components.extend(chain)
        chains_as_indexes.append(list(range(offset, offset + len(chain))))
        offset += len(chain)
    component_positions = {id(component): index for index, component in enumerate(flat_components)}
    degraded_indexes = {component_positions[component_id] for component_id in degraded_component_ids if component_id in component_positions}
    run_indexes = extract_qualifying_degraded_run_indexes(chains_as_indexes, flat_components, degraded_indexes, link_distance)
    return [[flat_components[index] for index in run] for run in run_indexes]


def extract_qualifying_degraded_run_indexes(
    dot_chains: Sequence[Sequence[int]],
    components: Sequence[DotComponent],
    degraded_component_indexes: Sequence[int] | set[int],
    link_distance: float,
) -> list[list[int]]:
    degraded_flags = [False] * len(components)
    for index in degraded_component_indexes:
        if 0 <= index < len(degraded_flags):
            degraded_flags[index] = True

    raw_runs: list[list[int]] = []
    for chain in dot_chains:
        current_run: list[int] = []
        for component_index in chain:
            if degraded_flags[component_index]:
                current_run.append(component_index)
                continue
            if current_run:
                raw_runs.append(current_run.copy())
                current_run.clear()
        if current_run:
            raw_runs.append(current_run.copy())

    merged_runs = merge_chain_endpoint_indexes(raw_runs, components, link_distance)
    qualifying_runs: list[list[int]] = []
    for run in merged_runs:
        if len(run) < MIN_CONTINUOUS_RED_POINTS:
            continue
        if is_text_like_label_cluster_from_indexes(run, components):
            continue
        qualifying_runs.append(run)
    return qualifying_runs


def is_text_like_label_cluster_from_indexes(
    cluster_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> bool:
    return is_text_like_label_cluster([components[index] for index in cluster_indexes])


def chain_sort_key_from_indexes(
    chain_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> tuple[float, float]:
    centers = [components[index].center for index in chain_indexes]
    min_x = min(point[0] for point in centers)
    mean_y = sum(point[1] for point in centers) / len(centers)
    return (min_x, mean_y)


def cluster_total_area_from_indexes(
    cluster_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> int:
    return sum(components[index].area for index in cluster_indexes)


def cluster_max_extent_from_indexes(
    cluster_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> float:
    centers = [components[index].center for index in cluster_indexes]
    if len(centers) < 2:
        return 0.0
    if np is not None:
        matrix = pairwise_distance_matrix(centers)
        if matrix:
            return max(max(row) for row in matrix if row)
    max_extent = 0.0
    for index, (x1, y1) in enumerate(centers):
        for x2, y2 in centers[index + 1:]:
            max_extent = max(max_extent, math.hypot(x2 - x1, y2 - y1))
    return max_extent

    raw_runs: list[list[DotComponent]] = []
    for chain in dot_chains:
        current_run: list[DotComponent] = []
        for component in chain:
            if id(component) in degraded_component_ids:
                current_run.append(component)
                continue
            if current_run:
                raw_runs.append(current_run.copy())
            current_run.clear()

        if current_run:
            raw_runs.append(current_run.copy())

    merged_runs = merge_chain_endpoints(raw_runs, link_distance)
    return [
        run
        for run in merged_runs
        if len(run) >= MIN_CONTINUOUS_RED_POINTS and not is_text_like_label_cluster(list(run))
    ]


def choose_best_red_clusters(
    adaptive_clusters: list[list[dict[str, object] | DotComponent]],
    bbox_clusters: list[list[dict[str, object] | DotComponent]],
) -> tuple[list[list[dict[str, object] | DotComponent]], str]:
    filtered_adaptive = [cluster for cluster in adaptive_clusters if not is_text_like_label_cluster(cluster)]
    filtered_bbox = [cluster for cluster in bbox_clusters if not is_text_like_label_cluster(cluster)]

    adaptive_score = red_cluster_strategy_score(filtered_adaptive)
    bbox_score = red_cluster_strategy_score(filtered_bbox)
    if bbox_score > adaptive_score:
        return filtered_bbox, "bbox_gap"
    return filtered_adaptive, "adaptive_center"


def red_cluster_strategy_score(clusters: list[list[dict[str, object] | DotComponent]]) -> tuple[int, int, float]:
    qualifying = [cluster for cluster in clusters if len(cluster) >= MIN_CONTINUOUS_RED_POINTS]
    if not qualifying:
        return (0, 0, 0.0)
    largest = max(len(cluster) for cluster in qualifying)
    qualifying_count = len(qualifying)
    total_extent = sum(cluster_max_extent(cluster) for cluster in qualifying)
    return (largest, qualifying_count, total_extent)


def cluster_components(
    components: Sequence[dict[str, object] | DotComponent],
    link_distance: float = CLUSTER_LINK_DISTANCE,
) -> list[list[dict[str, object] | DotComponent]]:
    if not components:
        return []

    centers = [component_center(component) for component in components]
    components = list(components)
    visited = [False] * len(components)
    clusters: list[list[dict[str, object] | DotComponent]] = []

    for index in range(len(components)):
        if visited[index]:
            continue

        stack = [index]
        visited[index] = True
        cluster: list[dict[str, object] | DotComponent] = []

        while stack:
            current = stack.pop()
            cluster.append(components[current])
            current_x, current_y = centers[current]

            for other_index in range(len(components)):
                if visited[other_index]:
                    continue
                other_x, other_y = centers[other_index]
                if math.hypot(current_x - other_x, current_y - other_y) > link_distance:
                    continue
                visited[other_index] = True
                stack.append(other_index)

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


def cluster_components_by_bbox_gap(
    components: Sequence[dict[str, object] | DotComponent],
    gap_distance: float = RED_POINT_GAP_DISTANCE,
) -> list[list[dict[str, object] | DotComponent]]:
    if not components:
        return []

    components = list(components)
    visited = [False] * len(components)
    boxes = [component_bbox(component) for component in components]
    clusters: list[list[dict[str, object] | DotComponent]] = []

    for index in range(len(components)):
        if visited[index]:
            continue

        stack = [index]
        visited[index] = True
        cluster: list[dict[str, object] | DotComponent] = []

        while stack:
            current = stack.pop()
            cluster.append(components[current])

            for other_index in range(len(components)):
                if visited[other_index]:
                    continue
                if component_bbox_gap_distance(boxes[current], boxes[other_index]) > gap_distance:
                    continue
                visited[other_index] = True
                stack.append(other_index)

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters




def component_bbox(component: dict[str, object] | DotComponent) -> tuple[int, int, int, int]:
    return build_component_like(component).bbox


def component_bbox_gap_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_min_x, left_min_y, left_max_x, left_max_y = left
    right_min_x, right_min_y, right_max_x, right_max_y = right
    dx = max(0, max(left_min_x - right_max_x, right_min_x - left_max_x))
    dy = max(0, max(left_min_y - right_max_y, right_min_y - left_max_y))
    return math.hypot(dx, dy)


def component_center(component: dict[str, object] | DotComponent) -> tuple[float, float]:
    return build_component_like(component).center


def cluster_sort_key(cluster: list[dict[str, object] | DotComponent]) -> tuple[float, float]:
    centers = [component_center(component) for component in cluster]
    min_x = min(point[0] for point in centers)
    mean_y = sum(point[1] for point in centers) / len(centers)
    return (min_x, mean_y)


def cluster_total_area(cluster: list[dict[str, object] | DotComponent]) -> int:
    return sum(build_component_like(component).area for component in cluster)


def cluster_max_extent(cluster: list[dict[str, object] | DotComponent]) -> float:
    centers = [component_center(component) for component in cluster]
    if len(centers) < 2:
        return 0.0
    if np is not None:
        matrix = pairwise_distance_matrix(centers)
        if matrix:
            return max(max(row) for row in matrix if row)
    max_extent = 0.0
    for index, (x1, y1) in enumerate(centers):
        for x2, y2 in centers[index + 1:]:
            max_extent = max(max_extent, math.hypot(x2 - x1, y2 - y1))
    return max_extent


def is_text_like_label_cluster(cluster: list[dict[str, object] | DotComponent]) -> bool:
    if len(cluster) < MIN_CONTINUOUS_RED_POINTS:
        return False

    min_x, min_y, max_x, max_y = cluster_bbox(cluster)
    cluster_height = max_y - min_y + 1
    if cluster_height > 32:
        return False

    centers = [component_center(component) for component in cluster]
    row_groups = group_cluster_rows(centers, tolerance=5.0)
    if len(row_groups) != 2:
        return False

    row_groups.sort(key=lambda group: statistics.mean(point[1] for point in group))
    upper_row, lower_row = row_groups
    if len(upper_row) < 2 or len(lower_row) < 2:
        return False

    upper_y = statistics.mean(point[1] for point in upper_row)
    lower_y = statistics.mean(point[1] for point in lower_row)
    row_gap = lower_y - upper_y
    if row_gap < 8.0 or row_gap > 24.0:
        return False

    upper_spread = max(abs(point[1] - upper_y) for point in upper_row)
    lower_spread = max(abs(point[1] - lower_y) for point in lower_row)
    if upper_spread > 4.0 or lower_spread > 4.0:
        return False

    upper_min_x = min(point[0] for point in upper_row)
    upper_max_x = max(point[0] for point in upper_row)
    lower_min_x = min(point[0] for point in lower_row)
    lower_max_x = max(point[0] for point in lower_row)
    overlap = min(upper_max_x, lower_max_x) - max(upper_min_x, lower_min_x)
    smaller_width = min(upper_max_x - upper_min_x, lower_max_x - lower_min_x)
    if smaller_width <= 0:
        return False

    overlap_ratio = overlap / smaller_width
    return overlap_ratio >= 0.45


def group_cluster_rows(
    centers: list[tuple[float, float]],
    tolerance: float,
) -> list[list[tuple[float, float]]]:
    groups: list[list[tuple[float, float]]] = []
    for center in sorted(centers, key=lambda point: point[1]):
        placed = False
        for group in groups:
            group_mean_y = statistics.mean(point[1] for point in group)
            if abs(center[1] - group_mean_y) <= tolerance:
                group.append(center)
                placed = True
                break
        if not placed:
            groups.append([center])
    return groups


def cluster_bbox(cluster: list[dict[str, object] | DotComponent]) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for component in cluster:
        for pixel_x, pixel_y in build_component_like(component).pixels:
            xs.append(pixel_x)
            ys.append(pixel_y)
    return min(xs), min(ys), max(xs), max(ys)


def hotspot_circle(components: list[dict[str, object] | DotComponent]) -> tuple[float, float, float] | None:
    if not components:
        return None

    pixels = [pixel for component in components for pixel in build_component_like(component).pixels]
    xs = [pixel[0] for pixel in pixels]
    ys = [pixel[1] for pixel in pixels]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    base_radius = max(math.hypot(pixel_x - center_x, pixel_y - center_y) for pixel_x, pixel_y in pixels)
    radius = max(18.0, base_radius + HOTSPOT_PADDING)
    return center_x, center_y, radius


def component_hotspot_circles(
    components: Sequence[dict[str, object] | DotComponent],
) -> list[tuple[float, float, float]]:
    circles: list[tuple[float, float, float]] = []
    for component in components:
        normalized = build_component_like(component)
        center_x, center_y = normalized.center
        local_radius = max(18.0, (max(normalized.width, normalized.height) * 0.72) + 10.0)
        circles.append((center_x, center_y, local_radius))
    return circles


def build_run_summary_from_indexes(
    run_indexes: Sequence[int],
    components: Sequence[DotComponent],
) -> dict[str, object]:
    run_components = [components[index] for index in run_indexes]
    circle = hotspot_circle(run_components)
    return {
        "indexes": list(run_indexes),
        "components": run_components,
        "total_area": cluster_total_area_from_indexes(run_indexes, components),
        "max_extent": cluster_max_extent_from_indexes(run_indexes, components),
        "sort_key": chain_sort_key_from_indexes(run_indexes, components),
        "circle": circle,
        "circles": [] if circle is None else [circle],
    }


def build_run_summary(
    run_components: Sequence[dict[str, object] | DotComponent],
    segmented: bool = False,
    bitmap: Bitmap | None = None,
) -> dict[str, object]:
    components = [build_component_like(component) for component in run_components]
    circle = hotspot_circle(list(components))
    annotation_components = components
    if segmented and bitmap is not None:
        filtered_annotation_components = [
            component
            for component in components
            if not component_has_bright_inner_symbol(bitmap, component)
        ]
        if filtered_annotation_components:
            annotation_components = filtered_annotation_components
    circles = component_hotspot_circles(annotation_components) if segmented else ([] if circle is None else [circle])
    return {
        "indexes": list(range(len(components))),
        "components": components,
        "total_area": cluster_total_area(list(components)),
        "max_extent": cluster_max_extent(list(components)),
        "sort_key": cluster_sort_key(list(components)),
        "circle": circle,
        "circles": circles,
    }


def nearest_neighbor_distances_for_centers(
    centers: Sequence[tuple[float, float]],
) -> list[float]:
    if len(centers) < 2:
        return []
    if np is not None:
        center_array = np.asarray(centers, dtype=np.float32)
        diff = center_array[:, None, :] - center_array[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        return nearest.astype(float).tolist()

    nearest_neighbor_distances: list[float] = []
    for index, (x1, y1) in enumerate(centers):
        distances = [
            math.hypot(x2 - x1, y2 - y1)
            for other_index, (x2, y2) in enumerate(centers)
            if other_index != index
        ]
        if distances:
            nearest_neighbor_distances.append(min(distances))
    return nearest_neighbor_distances


def pairwise_distance_matrix(
    centers: Sequence[tuple[float, float]],
) -> list[list[float]]:
    count = len(centers)
    if count == 0:
        return []
    if np is not None:
        center_array = np.asarray(centers, dtype=np.float32)
        diff = center_array[:, None, :] - center_array[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32))
        return distances.astype(float).tolist()

    matrix = [[math.inf] * count for _ in range(count)]
    for left_index in range(count):
        matrix[left_index][left_index] = 0.0
        left_x, left_y = centers[left_index]
        for right_index in range(left_index + 1, count):
            right_x, right_y = centers[right_index]
            distance = math.hypot(right_x - left_x, right_y - left_y)
            matrix[left_index][right_index] = distance
            matrix[right_index][left_index] = distance
    return matrix


def build_kpi_annotated_preview(
    bitmap: Bitmap,
    preview_image_uri: str,
    hotspot_circles: Sequence[tuple[float, float, float]],
) -> str:
    width = bitmap.width
    height = bitmap.height
    circle_markup = ""
    if hotspot_circles:
        scale = max(1.0, width / 1400.0)
        outer_stroke = min(10.0, 6.0 * scale)
        inner_stroke = min(5.0, 3.0 * scale)
        circles: list[str] = []
        for circle in hotspot_circles:
            center_x, center_y, radius = circle
            circles.append(
                f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{radius:.2f}" '
                f'fill="#ff7c87" fill-opacity="0.16" stroke="#fff4f4" stroke-width="{inner_stroke:.2f}" />'
                f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{radius:.2f}" '
                f'fill="none" stroke="#ff7684" stroke-width="{outer_stroke:.2f}" stroke-dasharray="10 7" />'
            )
        circle_markup = "".join(circles)

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="{preview_image_uri}" width="{width}" height="{height}" />
  {circle_markup}
</svg>
""".strip()

    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
