from __future__ import annotations

import base64
import logging
import math
import time
from typing import Any, Iterable

from .acceleration import (
    bitmap_hsv_array,
    bitmap_rgb_array,
    rgb_pixel,
    build_integral_image as build_integral_image_shared,
    extract_binary_components,
    hsv_pixel,
    neighborhood_sum as neighborhood_sum_shared,
    np,
)
from .models import AnalysisOutcome, Bitmap, DetectedColor

LOGGER = logging.getLogger(__name__)

LEGEND_X_RATIO = 0.22
LEGEND_Y_RATIO = 0.25
SITE_HISTOGRAM_X_RANGE = (0.45, 0.75)
SITE_HISTOGRAM_Y_RANGE = (0.10, 0.55)
SITE_HISTOGRAM_FALLBACK_X_RANGE = (0.05, 0.92)
SITE_HISTOGRAM_FALLBACK_Y_RANGE = (0.05, 0.92)
SITE_COLOR_SEARCH_X_RANGE = (0.35, 0.82)
SITE_COLOR_SEARCH_Y_RANGE = (0.10, 0.60)
SITE_COLOR_SEARCH_FALLBACK_X_RANGE = (0.05, 0.92)
SITE_COLOR_SEARCH_FALLBACK_Y_RANGE = (0.05, 0.92)
SITE_CENTER_WINDOW_RADIUS = 16
SITE_CENTER_MIN_DENSITY = 35
SITE_CENTER_SAMPLE_LIMIT = 90
SITE_CENTER_DISTANCE_WEIGHT = 2.5
SITE_CENTER_SCORE_FLOOR_RATIO = 0.82
SITE_FAN_INNER_RADIUS = 6
SITE_FAN_OUTER_RADIUS = 36
SITE_FAN_TARGET_RADIUS = 18
SITE_FAN_DENSITY_RADIUS = 5
SITE_FAN_MIN_DENSITY = 22
SECTOR_PROTOTYPE_HUE_WINDOW_DEG = 10.0
SECTOR_RGB_MATCH_CAP = 80.0
SECTOR_RGB_MATCH_FLOOR = 45.0
SECTOR_HUE_MATCH_CAP_DEG = 18.0
SECTOR_HUE_MATCH_FLOOR_DEG = 8.0
COLOR_SATURATION_THRESHOLD = 0.50
COLOR_VALUE_THRESHOLD = 0.35
DENSE_WINDOW_RADIUS = 7
DENSE_FOR_HUE_PEAKS = 15
DENSE_FOR_SITE_COLOR = 14
POINT_WINDOW_RADIUS = 3
POINT_MIN_DENSITY = 13
POINT_MAX_DENSITY = 45
POINT_MIN_RADIUS = 22
POINT_COMPONENT_MIN_PIXELS = 5
POINT_COMPONENT_MAX_PIXELS = 90
POINT_COMPONENT_MAX_SPAN = 14
POINT_COMPONENT_MIN_FILL = 0.2
SITE_ATTACHED_COMPONENT_MAX_DISTANCE = 40.0
SITE_ATTACHED_FALLBACK_MISMATCH_DEG = 50.0
POINT_COMPONENT_NEAR_SITE_RADIUS = 60.0
POINT_COMPONENT_NEAR_SITE_RATIO_MAX = 0.35
SITE_ATTACHED_NEAR_SITE_POINT_MAX_PIXELS = 40
SITE_ATTACHED_POINT_ANGLE_WINDOW_DEG = 45.0
SITE_ATTACHED_BRANCH_ANGLE_WINDOW_DEG = 60.0
BRANCH_COMPONENT_MIN_PIXELS = 70
BRANCH_COMPONENT_MIN_SPAN = 18
BRANCH_COMPONENT_MAX_FILL = 0.45
MIN_COLOR_PEAK_PIXELS = 30
MIN_POINT_PIXELS_PER_COLOR = 80
MIN_TOTAL_POINT_PIXELS = 300
LOW_RES_CROSS_MAX_WIDTH = 900
LOW_RES_CROSS_MAX_HEIGHT = 450
LOW_RES_SITE_WINDOW_RADIUS = 16
LOW_RES_MIN_COMPONENTS_PER_GROUP = 3
LOW_RES_MIN_GROUPS = 2
LOW_RES_EXCLUDE_NEAR_SITE_RADIUS = 25.0
LOW_RES_CROSS_THRESHOLD = 0.35
HUE_MATCH_FOR_SITE = 0.06
HUE_MATCH_FOR_POINTS = 0.08
MISASSIGN_MARGIN_DEG = 12.0
MISASSIGNED_THRESHOLD = 0.18
MIXED_BIN_THRESHOLD = 0.20
INTRUSION_THRESHOLD = 0.18
LATE_HO_OWN_ZONE_MAX = 0.12
LATE_HO_SOURCE_COUNT_RATIO_MAX = 0.85
LATE_HO_SOURCE_TO_TARGET_MIN = 0.25
LATE_HO_TARGET_OWN_MIN = 0.80
LATE_HO_TARGET_OTHER_MAX = 0.35
LATE_HO_TARGET_ANGLE_MIN = 60.0
PAIR_LATE_HO_MISASSIGNED_MAX = 0.08
PAIR_LATE_HO_MIXED_MAX = 0.20
PAIR_LATE_HO_INTRUSION_MAX = 0.24
PAIR_LATE_HO_SOURCE_OWN_MIN = 0.75
PAIR_LATE_HO_SOURCE_TO_TARGET_MIN = 0.15
PAIR_LATE_HO_SOURCE_OTHER_MAX = 0.05
PAIR_LATE_HO_TARGET_OWN_MIN = 0.90
PAIR_LATE_HO_TARGET_OTHER_MAX = 0.08
PAIR_LATE_HO_ANGLE_MIN = 90.0
MINOR_PAIR_HO_MISASSIGNED_MAX = 0.02
MINOR_PAIR_HO_MIXED_MAX = 0.05
MINOR_PAIR_HO_INTRUSION_MAX = 0.05
MINOR_PAIR_HO_SOURCE_TO_TARGET_MIN = 0.015
MINOR_PAIR_HO_SOURCE_TO_TARGET_MAX = 0.05
MINOR_PAIR_HO_TARGET_OWN_MIN = 0.95
MINOR_PAIR_HO_ANGLE_MIN = 90.0


class SsvAnalysisError(ValueError):
    """Raised when the extracted SSV image cannot be analyzed reliably."""

    def __init__(self, message: str, code: str | None = None, details: dict[str, object] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def analyze_bitmap(bitmap: Bitmap, preview_image_uri: str) -> AnalysisOutcome:
    stage_started = time.perf_counter()
    hsv_cache, colorful_mask = build_color_cache(bitmap)
    color_cache_s = time.perf_counter() - stage_started

    colorful_integral = build_integral_image(colorful_mask)
    try:
        return analyze_bitmap_dense(
            bitmap=bitmap,
            preview_image_uri=preview_image_uri,
            hsv_cache=hsv_cache,
            colorful_mask=colorful_mask,
            colorful_integral=colorful_integral,
            color_cache_s=color_cache_s,
        )
    except SsvAnalysisError as exc:
        fallback = analyze_bitmap_low_res_fallback(
            bitmap=bitmap,
            preview_image_uri=preview_image_uri,
            hsv_cache=hsv_cache,
            colorful_mask=colorful_mask,
            colorful_integral=colorful_integral,
            primary_reason=str(exc),
            primary_reason_code=getattr(exc, "code", None),
            color_cache_s=color_cache_s,
        )
        if fallback is not None:
            return fallback
        raise


def analyze_bitmap_dense(
    bitmap: Bitmap,
    preview_image_uri: str,
    hsv_cache: Any,
    colorful_mask: Any,
    colorful_integral: Any,
    color_cache_s: float,
) -> AnalysisOutcome:
    stage_started = time.perf_counter()
    site_center_hint = estimate_site_center_from_density(bitmap, colorful_mask, colorful_integral)
    site_hint_s = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    sector_hues, sector_hue_debug = detect_sector_hues(bitmap, hsv_cache, colorful_mask, colorful_integral, site_center_hint)
    sector_hues_s = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    site_center = estimate_site_center(bitmap, hsv_cache, colorful_mask, colorful_integral, sector_hues, site_center_hint)
    site_center_s = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    sector_signatures = extract_sector_signatures(bitmap, hsv_cache, colorful_mask, colorful_integral, sector_hues, site_center)
    sector_signatures_s = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    point_sets, evaluation_angles = segment_point_clouds(
        bitmap,
        hsv_cache,
        colorful_mask,
        colorful_integral,
        sector_hues,
        sector_signatures,
        site_center,
    )
    segment_point_clouds_s = time.perf_counter() - stage_started

    total_points = sum(len(point_set["angles"]) for point_set in point_sets)
    if total_points < MIN_TOTAL_POINT_PIXELS:
        raise SsvAnalysisError(
            "The extracted image does not contain enough detected serving points to evaluate crossing.",
            code="insufficient_total_points",
        )

    ordered_indices = sorted(range(len(point_sets)), key=lambda index: evaluation_angles[index])
    sector_boundaries = compute_sector_boundaries([evaluation_angles[index] for index in ordered_indices])

    ordered_colors: list[DetectedColor] = []
    ordered_point_sets = [point_sets[index] for index in ordered_indices]
    ordered_angles = [evaluation_angles[index] for index in ordered_indices]

    for output_index, original_index in enumerate(ordered_indices, start=1):
        point_set = point_sets[original_index]
        ordered_colors.append(
            DetectedColor(
                name=f"sector_{output_index}",
                rgb=point_set["rgb"],
                hex=rgb_to_hex(point_set["rgb"]),
                dominant_angle=evaluation_angles[original_index],
                point_count=len(point_set["angles"]),
                site_angle=sector_signatures[original_index]["site_angle"],
            )
        )

    stage_started = time.perf_counter()
    misassigned_ratio = compute_misassigned_ratio(ordered_angles, ordered_point_sets)
    mixed_bin_ratio = compute_mixed_bin_ratio(ordered_point_sets)
    intrusion_ratios = compute_intrusion_ratios(ordered_angles, sector_boundaries, ordered_point_sets)
    zone_matrix = compute_zone_matrix(sector_boundaries, ordered_point_sets)
    max_intrusion_ratio = max(intrusion_ratios)
    min_angle_separation = compute_min_angle_separation(ordered_angles)
    confidence = compute_confidence(total_points, min_angle_separation, misassigned_ratio, mixed_bin_ratio, max_intrusion_ratio)

    cross = (
        misassigned_ratio >= MISASSIGNED_THRESHOLD
        or mixed_bin_ratio >= MIXED_BIN_THRESHOLD
        or max_intrusion_ratio >= INTRUSION_THRESHOLD
    )
    warning_details = detect_late_ho_warnings(ordered_colors, ordered_point_sets, zone_matrix)
    if not warning_details:
        warning_details = detect_pair_late_ho_warnings(
            detected_colors=ordered_colors,
            point_sets=ordered_point_sets,
            zone_matrix=zone_matrix,
            intrusion_ratios=intrusion_ratios,
            misassigned_ratio=misassigned_ratio,
            mixed_bin_ratio=mixed_bin_ratio,
        )
    if not warning_details:
        warning_details = detect_minor_pair_late_ho_warnings(
            detected_colors=ordered_colors,
            point_sets=ordered_point_sets,
            zone_matrix=zone_matrix,
            intrusion_ratios=intrusion_ratios,
            misassigned_ratio=misassigned_ratio,
            mixed_bin_ratio=mixed_bin_ratio,
        )
    warnings = [warning_detail["message"] for warning_detail in warning_details]
    if cross and warning_details:
        cross = False
    verdict = "Cross detected" if cross else "No cross detected"
    cross_metrics_s = time.perf_counter() - stage_started

    metrics = {
        "mixed_bin_ratio": round(mixed_bin_ratio, 4),
        "misassigned_pixel_ratio": round(misassigned_ratio, 4),
        "max_intrusion_ratio": round(max_intrusion_ratio, 4),
        "intrusion_ratio_by_sector": {
            f"sector_{index + 1}": round(intrusion_ratios[index], 4)
            for index in range(len(intrusion_ratios))
        },
        "dominant_angles": {
            f"sector_{index + 1}": round(ordered_angles[index], 2)
            for index in range(len(ordered_angles))
        },
        "total_point_pixels": total_points,
        "min_angle_separation": round(min_angle_separation, 2),
        "confidence": round(confidence, 4),
        "analysis_mode": "dense_cross_solver",
        "dense_debug": sector_hue_debug,
        "warnings": warnings,
        "stage_timings": {
            "color_cache_s": round(color_cache_s, 4),
            "site_hint_s": round(site_hint_s, 4),
            "sector_hues_s": round(sector_hues_s, 4),
            "site_center_s": round(site_center_s, 4),
            "sector_signatures_s": round(sector_signatures_s, 4),
            "segment_point_clouds_s": round(segment_point_clouds_s, 4),
            "cross_metrics_s": round(cross_metrics_s, 4),
        },
    }

    stage_started = time.perf_counter()
    annotated_preview = build_annotated_preview(
        bitmap=bitmap,
        preview_image_uri=preview_image_uri,
        site_center=site_center,
        detected_colors=ordered_colors,
        sector_boundaries=sector_boundaries,
        verdict=verdict,
        metrics=metrics,
    )
    metrics["stage_timings"]["annotation_s"] = round(time.perf_counter() - stage_started, 4)

    return AnalysisOutcome(
        cross=cross,
        verdict=verdict,
        detected_colors=ordered_colors,
        metrics=metrics,
        site_center={"x": round(site_center[0], 2), "y": round(site_center[1], 2)},
        annotated_preview=annotated_preview,
        warnings=warnings,
        warning_details=warning_details,
    )


def analyze_bitmap_low_res_fallback(
    bitmap: Bitmap,
    preview_image_uri: str,
    hsv_cache: Any,
    colorful_mask: Any,
    colorful_integral: Any,
    primary_reason: str,
    primary_reason_code: str | None,
    color_cache_s: float,
) -> AnalysisOutcome | None:
    if bitmap.width > LOW_RES_CROSS_MAX_WIDTH or bitmap.height > LOW_RES_CROSS_MAX_HEIGHT:
        return None

    stage_started = time.perf_counter()
    site_center = estimate_low_res_site_center(bitmap, colorful_mask, colorful_integral)
    if site_center is None:
        return None
    site_hint_s = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    sector_groups = build_low_res_sector_groups(bitmap, site_center, colorful_mask)
    grouping_s = time.perf_counter() - stage_started
    if len(sector_groups) < LOW_RES_MIN_GROUPS:
        return None
    display_groups = infer_low_res_display_groups(bitmap, site_center, colorful_mask, sector_groups)

    ordered_groups = sorted(sector_groups, key=lambda item: item["dominant_angle"])
    ordered_colors: list[DetectedColor] = []
    ordered_point_sets: list[dict[str, object]] = []
    ordered_angles: list[float] = []

    for output_index, group in enumerate(ordered_groups, start=1):
        point_set = {
            "angles": group["angles"],
            "rgb_samples": group["rgb_samples"],
            "rgb": average_rgb(group["rgb_samples"]),
        }
        ordered_point_sets.append(point_set)
        ordered_angles.append(group["dominant_angle"])
        ordered_colors.append(
            DetectedColor(
                name=f"sector_{output_index}",
                rgb=point_set["rgb"],
                hex=rgb_to_hex(point_set["rgb"]),
                dominant_angle=group["dominant_angle"],
                point_count=len(group["angles"]),
                site_angle=group["dominant_angle"],
            )
        )

    display_groups_sorted = sorted(display_groups, key=lambda item: item["dominant_angle"]) if display_groups else ordered_groups
    display_colors = [
        DetectedColor(
            name=f"sector_{index + 1}",
            rgb=average_rgb(group["rgb_samples"]),
            hex=rgb_to_hex(average_rgb(group["rgb_samples"])),
            dominant_angle=group["dominant_angle"],
            point_count=len(group["angles"]),
            site_angle=group["dominant_angle"],
        )
        for index, group in enumerate(display_groups_sorted)
    ]
    display_boundaries = compute_sector_boundaries([group["dominant_angle"] for group in display_groups_sorted])

    total_points = sum(len(point_set["angles"]) for point_set in ordered_point_sets)
    sector_boundaries = compute_sector_boundaries(ordered_angles)
    misassigned_ratio = compute_misassigned_ratio(ordered_angles, ordered_point_sets)
    mixed_bin_ratio = compute_mixed_bin_ratio(ordered_point_sets)
    intrusion_ratios = compute_intrusion_ratios(ordered_angles, sector_boundaries, ordered_point_sets)
    zone_matrix = compute_zone_matrix(sector_boundaries, ordered_point_sets)
    max_intrusion_ratio = max(intrusion_ratios) if intrusion_ratios else 0.0
    min_angle_separation = compute_min_angle_separation(ordered_angles)
    confidence = compute_confidence(
        total_points=max(total_points * 12, total_points),
        min_angle_separation=min_angle_separation,
        misassigned_ratio=misassigned_ratio,
        mixed_bin_ratio=mixed_bin_ratio,
        intrusion_ratio=max_intrusion_ratio,
    )

    cross = (
        misassigned_ratio >= LOW_RES_CROSS_THRESHOLD
        or mixed_bin_ratio >= LOW_RES_CROSS_THRESHOLD
        or max_intrusion_ratio >= LOW_RES_CROSS_THRESHOLD
    )
    verdict = "Cross detected" if cross else "No cross detected"

    metrics = {
        "mixed_bin_ratio": round(mixed_bin_ratio, 4),
        "misassigned_pixel_ratio": round(misassigned_ratio, 4),
        "max_intrusion_ratio": round(max_intrusion_ratio, 4),
        "intrusion_ratio_by_sector": {
            f"sector_{index + 1}": round(intrusion_ratios[index], 4)
            for index in range(len(intrusion_ratios))
        },
        "dominant_angles": {
            f"sector_{index + 1}": round(ordered_angles[index], 2)
            for index in range(len(ordered_angles))
        },
        "total_point_pixels": total_points,
        "min_angle_separation": round(min_angle_separation, 2),
        "confidence": round(confidence, 4),
        "analysis_mode": "low_res_serving_pci_fallback",
        "analysis_reason": primary_reason,
        "analysis_reason_code": primary_reason_code,
        "low_res_group_count": len(ordered_groups),
        "display_sector_count": len(display_colors),
        "inferred_sector_count": len([group for group in display_groups_sorted if group.get("inferred")]),
        "stage_timings": {
            "color_cache_s": round(color_cache_s, 4),
            "site_hint_s": round(site_hint_s, 4),
            "low_res_grouping_s": round(grouping_s, 4),
        },
    }

    stage_started = time.perf_counter()
    annotated_preview = build_annotated_preview(
        bitmap=bitmap,
        preview_image_uri=preview_image_uri,
        site_center=site_center,
        detected_colors=display_colors,
        sector_boundaries=display_boundaries,
        verdict=verdict,
        metrics=metrics,
    )
    metrics["stage_timings"]["annotation_s"] = round(time.perf_counter() - stage_started, 4)

    return AnalysisOutcome(
        cross=cross,
        verdict=verdict,
        detected_colors=display_colors,
        metrics=metrics,
        site_center={"x": round(site_center[0], 2), "y": round(site_center[1], 2)},
        annotated_preview=annotated_preview,
        warnings=[],
        warning_details=[],
    )


def estimate_low_res_site_center(
    bitmap: Bitmap,
    colorful_mask: Any,
    colorful_integral: Any,
) -> tuple[float, float] | None:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    candidates: list[tuple[int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if x < legend_x and y < legend_y:
                continue
            if not colorful_mask[y][x]:
                continue
            density = neighborhood_sum(colorful_integral, x, y, LOW_RES_SITE_WINDOW_RADIUS, width, height)
            if density < SITE_CENTER_MIN_DENSITY:
                continue
            candidates.append((density, x, y))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_density = candidates[0][0]
    selected = [candidate for candidate in candidates if candidate[0] >= (best_density * 0.90)][:18]
    if not selected:
        return None

    total_weight = sum(candidate[0] for candidate in selected)
    site_x = sum(candidate[0] * candidate[1] for candidate in selected) / total_weight
    site_y = sum(candidate[0] * candidate[2] for candidate in selected) / total_weight
    LOGGER.info("Estimated low-res site center: (%.2f, %.2f)", site_x, site_y)
    return site_x, site_y


def dense_image_scale(bitmap: Bitmap) -> float:
    return max(1.0, math.hypot(bitmap.width, bitmap.height) / 1000.0)


def scaled_radius(base_radius: int, scale: float, minimum: int) -> int:
    return max(minimum, int(round(base_radius * scale)))


def build_low_res_sector_groups(
    bitmap: Bitmap,
    site_center: tuple[float, float],
    colorful_mask: Any,
) -> list[dict[str, object]]:
    grouped_components = collect_low_res_grouped_components(bitmap, site_center, colorful_mask)

    groups: list[dict[str, object]] = []
    for bin_index, grouped in grouped_components.items():
        if len(grouped) < LOW_RES_MIN_COMPONENTS_PER_GROUP:
            continue
        selected = select_largest_angular_cluster(grouped)
        if len(selected) < LOW_RES_MIN_COMPONENTS_PER_GROUP:
            continue

        angles = [entry["angle"] for entry in selected]
        rgb_samples = [entry["mean_rgb"] for entry in selected]
        groups.append(
            {
                "angles": angles,
                "rgb_samples": rgb_samples,
                "dominant_angle": circular_mean_degrees(angles),
                "component_count": len(selected),
                "bin_index": bin_index,
            }
        )

    groups.sort(key=lambda item: (-item["component_count"], item["dominant_angle"]))
    return groups[:3]


def infer_low_res_display_groups(
    bitmap: Bitmap,
    site_center: tuple[float, float],
    colorful_mask: Any,
    strong_groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(strong_groups) >= 3:
        return sorted(strong_groups, key=lambda item: item["dominant_angle"])

    grouped_components = collect_low_res_grouped_components(bitmap, site_center, colorful_mask)
    legend_swatches = detect_cross_legend_swatches(bitmap)
    strong_bins = {group["bin_index"] for group in strong_groups if "bin_index" in group}
    strong_rgbs = [average_rgb(group["rgb_samples"]) for group in strong_groups]
    strong_angles = [group["dominant_angle"] for group in strong_groups]
    best_candidate: dict[str, object] | None = None

    for bin_index, grouped in grouped_components.items():
        if bin_index in strong_bins:
            continue
        selected = select_largest_angular_cluster(grouped)
        if not selected:
            continue
        angles = [entry["angle"] for entry in selected]
        dominant_angle = circular_mean_degrees(angles)
        if strong_angles and min(angular_distance_degrees(dominant_angle, angle) for angle in strong_angles) < 20.0:
            continue
        candidate = {
            "angles": angles,
            "rgb_samples": [entry["mean_rgb"] for entry in selected],
            "dominant_angle": dominant_angle,
            "component_count": len(selected),
            "bin_index": bin_index,
            "inferred": True,
        }
        if best_candidate is None or candidate["component_count"] > best_candidate["component_count"]:
            best_candidate = candidate

    display_groups = list(strong_groups)
    if best_candidate is not None:
        legend_rgb = select_missing_legend_swatch(legend_swatches, strong_rgbs, average_rgb(best_candidate["rgb_samples"]))
        if legend_rgb is not None:
            best_candidate["rgb_samples"] = [legend_rgb]
            best_candidate["legend_rgb"] = legend_rgb
        display_groups.append(best_candidate)
    return sorted(display_groups, key=lambda item: item["dominant_angle"])


def collect_low_res_grouped_components(
    bitmap: Bitmap,
    site_center: tuple[float, float],
    colorful_mask: Any,
) -> dict[int, list[dict[str, object]]]:
    components = extract_components(colorful_mask)
    grouped_components: dict[int, list[dict[str, object]]] = {}

    for component in components:
        if not is_low_res_cross_point_component(bitmap, component):
            continue
        if component_min_distance(component, site_center) <= LOW_RES_EXCLUDE_NEAR_SITE_RADIUS:
            continue

        mean_rgb = component_mean_rgb(bitmap, component)
        hue, saturation, value = rgb_to_hsv(*mean_rgb)
        hue_degrees = hue * 360.0
        if saturation < COLOR_SATURATION_THRESHOLD or value < COLOR_VALUE_THRESHOLD:
            continue
        if 20.0 <= hue_degrees < 80.0:
            continue

        bin_index = hue_bin_index(hue_degrees)
        grouped_components.setdefault(bin_index, []).append(
            {
                "component": component,
                "mean_rgb": mean_rgb,
                "angle": component_center_angle(component, site_center),
            }
        )
    return grouped_components


def detect_cross_legend_swatches(bitmap: Bitmap) -> list[tuple[int, int, int]]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    mask = [[0] * legend_x for _ in range(legend_y)]

    for y in range(legend_y):
        for x in range(legend_x):
            red, green, blue = rgb_pixel(bitmap, x, y)
            hue, saturation, value = rgb_to_hsv(red, green, blue)
            if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                mask[y][x] = 1

    swatches: list[tuple[int, int, int]] = []
    for component in extract_components(mask):
        area = int(component["area"])
        comp_width = int(component["width"])
        comp_height = int(component["height"])
        fill_ratio = area / max(comp_width * comp_height, 1)
        if area < 18 or area > 200:
            continue
        if comp_width < 4 or comp_height < 4:
            continue
        if fill_ratio < 0.6:
            continue
        swatches.append(component_mean_rgb(bitmap, component))

    unique_swatches: list[tuple[int, int, int]] = []
    for rgb in swatches:
        if any(rgb_distance(rgb, existing) < 35.0 for existing in unique_swatches):
            continue
        unique_swatches.append(rgb)
    return unique_swatches


def select_missing_legend_swatch(
    legend_swatches: list[tuple[int, int, int]],
    strong_rgbs: list[tuple[int, int, int]],
    candidate_rgb: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    unmatched = [
        swatch
        for swatch in legend_swatches
        if all(rgb_distance(swatch, strong_rgb) >= 60.0 for strong_rgb in strong_rgbs)
    ]
    if not unmatched:
        return None
    return min(unmatched, key=lambda swatch: rgb_distance(swatch, candidate_rgb))


def is_low_res_cross_point_component(bitmap: Bitmap, component: dict[str, object]) -> bool:
    if not is_point_like_component(component):
        return False

    area = int(component["area"])
    width = int(component["width"])
    height = int(component["height"])
    fill_ratio = area / max(width * height, 1)
    if fill_ratio < 0.35:
        return False
    if abs(width - height) > 4:
        return False
    if component_inner_bright_ratio(bitmap, component) > 0.22:
        return False
    return True


def component_mean_rgb(bitmap: Bitmap, component: dict[str, object]) -> tuple[int, int, int]:
    samples = [rgb_pixel(bitmap, x, y) for x, y in component["pixels"]]
    return average_rgb(samples)


def component_center_angle(component: dict[str, object], site_center: tuple[float, float]) -> float:
    min_x, min_y, max_x, max_y = component["bbox"]
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    site_x, site_y = site_center
    return angle_from_center(center_x - site_x, center_y - site_y)


def component_inner_bright_ratio(bitmap: Bitmap, component: dict[str, object]) -> float:
    min_x, min_y, max_x, max_y = component["bbox"]
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width < 4 or height < 4:
        return 0.0

    inset_x = max(1, width // 4)
    inset_y = max(1, height // 4)
    start_x = min_x + inset_x
    end_x = max_x - inset_x
    start_y = min_y + inset_y
    end_y = max_y - inset_y
    if start_x > end_x or start_y > end_y:
        return 0.0

    bright_pixels = 0
    total_pixels = 0
    for y in range(start_y, end_y + 1):
        for x in range(start_x, end_x + 1):
            red, green, blue = rgb_pixel(bitmap, x, y)
            total_pixels += 1
            if red >= 225 and green >= 225 and blue >= 225:
                bright_pixels += 1
    if total_pixels == 0:
        return 0.0
    return bright_pixels / total_pixels


def hue_bin_index(hue_degrees: float) -> int:
    return int(((hue_degrees + 15.0) % 360.0) // 30.0)


def select_largest_angular_cluster(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(entries) <= LOW_RES_MIN_COMPONENTS_PER_GROUP:
        return entries

    sorted_entries = sorted(entries, key=lambda item: item["angle"])
    clusters: list[list[dict[str, object]]] = []
    current_cluster = [sorted_entries[0]]

    for entry in sorted_entries[1:]:
        if angular_distance_degrees(entry["angle"], current_cluster[-1]["angle"]) <= 40.0:
            current_cluster.append(entry)
            continue
        clusters.append(current_cluster)
        current_cluster = [entry]
    clusters.append(current_cluster)

    if len(clusters) >= 2:
        first_angle = clusters[0][0]["angle"]
        last_angle = clusters[-1][-1]["angle"]
        wrap_gap = (first_angle + 360.0) - last_angle
        if wrap_gap <= 40.0:
            merged = clusters[-1] + clusters[0]
            clusters = [merged] + clusters[1:-1]

    clusters.sort(
        key=lambda cluster: (
            -len(cluster),
            -angular_cluster_span(cluster),
        )
    )
    return clusters[0]


def angular_cluster_span(cluster: list[dict[str, object]]) -> float:
    if len(cluster) <= 1:
        return 0.0
    ordered_angles = sorted(item["angle"] for item in cluster)
    return ordered_angles[-1] - ordered_angles[0]


def build_color_cache(bitmap: Bitmap) -> tuple[Any, Any]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)

    hsv_array = bitmap_hsv_array(bitmap)
    if hsv_array is not None:
        colorful_mask = (
            (hsv_array[..., 1] >= COLOR_SATURATION_THRESHOLD)
            & (hsv_array[..., 2] >= COLOR_VALUE_THRESHOLD)
        ).astype(np.uint8)
        colorful_mask[:legend_y, :legend_x] = 0
        return hsv_array, colorful_mask

    hsv_cache: list[list[tuple[float, float, float] | None]] = [[None] * width for _ in range(height)]
    colorful_mask = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            red, green, blue = rgb_pixel(bitmap, x, y)
            hue, saturation, value = rgb_to_hsv(red, green, blue)
            hsv_cache[y][x] = (hue, saturation, value)

            if x < legend_x and y < legend_y:
                continue

            if saturation >= COLOR_SATURATION_THRESHOLD and value >= COLOR_VALUE_THRESHOLD:
                colorful_mask[y][x] = 1

    return hsv_cache, colorful_mask


def estimate_site_center_from_density(
    bitmap: Bitmap,
    colorful_mask: list[list[int]],
    colorful_integral: list[list[int]],
) -> tuple[float, float]:
    width = bitmap.width
    height = bitmap.height
    prior_x = width * 0.60
    prior_y = height * 0.32
    candidates = collect_site_center_candidates(
        colorful_mask=colorful_mask,
        colorful_integral=colorful_integral,
        width=width,
        height=height,
        prior_x=prior_x,
        prior_y=prior_y,
        x_range=SITE_HISTOGRAM_X_RANGE,
        y_range=SITE_HISTOGRAM_Y_RANGE,
    )
    if not candidates:
        candidates = collect_site_center_candidates(
            colorful_mask=colorful_mask,
            colorful_integral=colorful_integral,
            width=width,
            height=height,
            prior_x=prior_x,
            prior_y=prior_y,
            x_range=SITE_HISTOGRAM_FALLBACK_X_RANGE,
            y_range=SITE_HISTOGRAM_FALLBACK_Y_RANGE,
        )
    if not candidates:
        raise SsvAnalysisError(
            "The serving-site center could not be estimated from the extracted image.",
            code="insufficient_site_center_density",
            details={
                "primary_search_x_range": SITE_HISTOGRAM_X_RANGE,
                "primary_search_y_range": SITE_HISTOGRAM_Y_RANGE,
                "fallback_search_x_range": SITE_HISTOGRAM_FALLBACK_X_RANGE,
                "fallback_search_y_range": SITE_HISTOGRAM_FALLBACK_Y_RANGE,
            },
        )

    candidates.sort(key=lambda item: (-item[0], item[2]))
    score_floor = candidates[0][0] * SITE_CENTER_SCORE_FLOOR_RATIO
    selected = [candidate for candidate in candidates if candidate[0] >= score_floor][:SITE_CENTER_SAMPLE_LIMIT]
    if not selected:
        raise SsvAnalysisError(
            "The serving-site center could not be estimated from the extracted image.",
            code="insufficient_site_center_density",
        )

    total_weight = sum(max(candidate[0], 1.0) for candidate in selected)
    site_x = sum(max(candidate[0], 1.0) * candidate[3] for candidate in selected) / total_weight
    site_y = sum(max(candidate[0], 1.0) * candidate[4] for candidate in selected) / total_weight
    LOGGER.info("Estimated coarse site center: (%.2f, %.2f)", site_x, site_y)
    return site_x, site_y


def collect_site_center_candidates(
    colorful_mask: Any,
    colorful_integral: Any,
    width: int,
    height: int,
    prior_x: float,
    prior_y: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> list[tuple[float, int, float, int, int]]:
    x_start = int(width * x_range[0])
    x_end = int(width * x_range[1])
    y_start = int(height * y_range[0])
    y_end = int(height * y_range[1])

    candidates: list[tuple[float, int, float, int, int]] = []
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if not colorful_mask[y][x]:
                continue

            density = neighborhood_sum(colorful_integral, x, y, SITE_CENTER_WINDOW_RADIUS, width, height)
            if density < SITE_CENTER_MIN_DENSITY:
                continue

            distance_to_prior = math.sqrt(((x - prior_x) ** 2) + ((y - prior_y) ** 2))
            score = density - (distance_to_prior * SITE_CENTER_DISTANCE_WEIGHT)
            candidates.append((score, density, distance_to_prior, x, y))
    return candidates


def detect_sector_hues(
    bitmap: Bitmap,
    hsv_cache: list[list[tuple[float, float, float] | None]],
    colorful_mask: list[list[int]],
    colorful_integral: list[list[int]],
    site_center: tuple[float, float],
) -> tuple[list[float], dict[str, object]]:
    hist = [0.0] * 36
    width = bitmap.width
    height = bitmap.height
    site_x, site_y = site_center
    scale = dense_image_scale(bitmap)
    fan_inner = scaled_radius(SITE_FAN_INNER_RADIUS, scale, SITE_FAN_INNER_RADIUS)
    fan_outer = scaled_radius(SITE_FAN_OUTER_RADIUS, scale, SITE_FAN_OUTER_RADIUS)
    fan_target = scaled_radius(SITE_FAN_TARGET_RADIUS, scale, SITE_FAN_TARGET_RADIUS)
    fan_density_radius = scaled_radius(SITE_FAN_DENSITY_RADIUS, scale, SITE_FAN_DENSITY_RADIUS)
    x_start = max(0, int(site_x - fan_outer))
    x_end = min(width, int(site_x + fan_outer + 1))
    y_start = max(0, int(site_y - fan_outer))
    y_end = min(height, int(site_y + fan_outer + 1))

    peaks: list[int] = []
    fan_pixels = 0
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if not colorful_mask[y][x]:
                continue

            dx = x - site_x
            dy = y - site_y
            radius = math.sqrt((dx * dx) + (dy * dy))
            if radius < fan_inner or radius > fan_outer:
                continue

            density = neighborhood_sum(colorful_integral, x, y, fan_density_radius, width, height)
            if density < SITE_FAN_MIN_DENSITY:
                continue

            hue = hsv_pixel(hsv_cache, x, y)[0]
            radial_weight = max(0.2, 1.0 - (abs(radius - fan_target) / fan_outer))
            hist[int(hue * 36) % 36] += density * radial_weight
            fan_pixels += 1

    for count, index in sorted(((value, idx) for idx, value in enumerate(hist)), reverse=True):
        if count < float(MIN_COLOR_PEAK_PIXELS * 6):
            continue
        if any(circular_bin_distance(index, chosen) <= 2 for chosen in peaks):
            continue
        peaks.append(index)
        if len(peaks) == 3:
            break

    used_fallback = False
    if len(peaks) < 3:
        peaks = detect_sector_hues_fallback(bitmap, hsv_cache, colorful_mask, colorful_integral)
        used_fallback = True

    LOGGER.info("Detected sector hue peaks: %s", peaks)
    return (
        [((peak + 0.5) / 36.0) for peak in peaks],
        {
            "fan_pixels": fan_pixels,
            "fan_hist_max": round(max(hist) if hist else 0.0, 2),
            "sector_hue_count": len(peaks),
            "used_histogram_fallback": used_fallback,
            "fan_inner_radius": fan_inner,
            "fan_outer_radius": fan_outer,
            "fan_density_radius": fan_density_radius,
            "scale_factor": round(scale, 3),
        },
    )


def detect_sector_hues_fallback(
    bitmap: Bitmap,
    hsv_cache: list[list[tuple[float, float, float] | None]],
    colorful_mask: list[list[int]],
    colorful_integral: list[list[int]],
) -> list[int]:
    width = bitmap.width
    height = bitmap.height
    hist = [0] * 36
    x_start = int(width * SITE_HISTOGRAM_X_RANGE[0])
    x_end = int(width * SITE_HISTOGRAM_X_RANGE[1])
    y_start = int(height * SITE_HISTOGRAM_Y_RANGE[0])
    y_end = int(height * SITE_HISTOGRAM_Y_RANGE[1])

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if not colorful_mask[y][x]:
                continue
            density = neighborhood_sum(colorful_integral, x, y, DENSE_WINDOW_RADIUS, width, height)
            if density < DENSE_FOR_HUE_PEAKS:
                continue
            hue = hsv_pixel(hsv_cache, x, y)[0]
            hist[int(hue * 36) % 36] += 1

    peaks: list[int] = []
    for count, index in sorted(((value, idx) for idx, value in enumerate(hist)), reverse=True):
        if count < MIN_COLOR_PEAK_PIXELS:
            continue
        if any(circular_bin_distance(index, chosen) <= 2 for chosen in peaks):
            continue
        peaks.append(index)
        if len(peaks) == 3:
            break

    if len(peaks) < 3:
        raise SsvAnalysisError(
            "The serving-sector colors could not be identified from the extracted image.",
            code="insufficient_sector_colors",
            details={"peak_count": len(peaks), "peak_threshold": MIN_COLOR_PEAK_PIXELS},
        )

    return peaks


def estimate_site_center(
    bitmap: Bitmap,
    hsv_cache: list[list[tuple[float, float, float] | None]],
    colorful_mask: list[list[int]],
    colorful_integral: list[list[int]],
    sector_hues: list[float],
    site_center_hint: tuple[float, float],
) -> tuple[float, float]:
    width = bitmap.width
    height = bitmap.height
    prior_x, prior_y = site_center_hint
    scale = dense_image_scale(bitmap)
    fan_outer = scaled_radius(SITE_FAN_OUTER_RADIUS, scale, SITE_FAN_OUTER_RADIUS)
    color_centers = collect_sector_color_centers(
        bitmap=bitmap,
        hsv_cache=hsv_cache,
        colorful_mask=colorful_mask,
        colorful_integral=colorful_integral,
        sector_hues=sector_hues,
        prior_x=prior_x,
        prior_y=prior_y,
        fan_outer=fan_outer,
        x_range=SITE_COLOR_SEARCH_X_RANGE,
        y_range=SITE_COLOR_SEARCH_Y_RANGE,
    )
    if len(color_centers) < 3:
        color_centers = collect_sector_color_centers(
            bitmap=bitmap,
            hsv_cache=hsv_cache,
            colorful_mask=colorful_mask,
            colorful_integral=colorful_integral,
            sector_hues=sector_hues,
            prior_x=prior_x,
            prior_y=prior_y,
            fan_outer=fan_outer,
            x_range=SITE_COLOR_SEARCH_FALLBACK_X_RANGE,
            y_range=SITE_COLOR_SEARCH_FALLBACK_Y_RANGE,
        )

    if len(color_centers) < 3:
        return site_center_hint

    site_x = sum(center[0] for center in color_centers) / len(color_centers)
    site_y = sum(center[1] for center in color_centers) / len(color_centers)
    LOGGER.info("Estimated site center: (%.2f, %.2f)", site_x, site_y)
    return site_x, site_y


def collect_sector_color_centers(
    bitmap: Bitmap,
    hsv_cache: Any,
    colorful_mask: Any,
    colorful_integral: Any,
    sector_hues: list[float],
    prior_x: float,
    prior_y: float,
    fan_outer: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> list[tuple[float, float]]:
    width = bitmap.width
    height = bitmap.height
    x_start = int(width * x_range[0])
    x_end = int(width * x_range[1])
    y_start = int(height * y_range[0])
    y_end = int(height * y_range[1])

    color_centers: list[tuple[float, float]] = []
    for sector_hue in sector_hues:
        weighted_pixels: list[tuple[float, int, int]] = []
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                if not colorful_mask[y][x]:
                    continue

                density = neighborhood_sum(colorful_integral, x, y, DENSE_WINDOW_RADIUS, width, height)
                if density < DENSE_FOR_SITE_COLOR:
                    continue

                hue = hsv_pixel(hsv_cache, x, y)[0]
                if circular_hue_distance(hue, sector_hue) >= HUE_MATCH_FOR_SITE:
                    continue

                distance = ((x - prior_x) ** 2) + ((y - prior_y) ** 2)
                if distance > (fan_outer * fan_outer * 2.25):
                    continue
                weighted_pixels.append((distance, x, y))

        weighted_pixels.sort(key=lambda item: item[0])
        if not weighted_pixels:
            continue

        closest_pixels = weighted_pixels[:40]
        color_centers.append(
            (
                sum(item[1] for item in closest_pixels) / len(closest_pixels),
                sum(item[2] for item in closest_pixels) / len(closest_pixels),
            )
        )
    return color_centers


def segment_point_clouds(
    bitmap: Bitmap,
    hsv_cache: list[list[tuple[float, float, float] | None]],
    colorful_mask: list[list[int]],
    colorful_integral: list[list[int]],
    sector_hues: list[float],
    sector_signatures: list[dict[str, object]],
    site_center: tuple[float, float],
) -> tuple[list[dict[str, object]], list[float]]:
    width = bitmap.width
    height = bitmap.height
    legend_x = int(width * LEGEND_X_RATIO)
    legend_y = int(height * LEGEND_Y_RATIO)
    site_x, site_y = site_center
    sector_prototypes = [signature["rgb"] for signature in sector_signatures]
    use_array_masks = np is not None and isinstance(colorful_mask, np.ndarray)
    sector_masks = (
        [np.zeros((height, width), dtype=np.uint8) for _ in sector_hues]
        if use_array_masks
        else [[[0] * width for _ in range(height)] for _ in sector_hues]
    )
    rgb_array = bitmap_rgb_array(bitmap) if use_array_masks else None

    point_sets = [
        {
            "angles": [],
            "rgb_samples": [],
        }
        for _ in sector_hues
    ]
    evaluation_angles: list[float] = []

    if use_array_masks:
        candidate_pixels = np.argwhere(colorful_mask != 0)
        for y_value, x_value in candidate_pixels.tolist():
            x = int(x_value)
            y = int(y_value)
            density = neighborhood_sum(colorful_integral, x, y, POINT_WINDOW_RADIUS, width, height)
            if density < POINT_MIN_DENSITY or density > POINT_MAX_DENSITY:
                continue

            hue = float(hsv_cache[y, x, 0])
            hue_distances = [circular_hue_distance(hue, sector_hue) * 360.0 for sector_hue in sector_hues]
            pixel = rgb_array[y, x]
            pixel_rgb = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
            rgb_distances = [rgb_distance(pixel_rgb, prototype) for prototype in sector_prototypes]
            sector_index = min(
                range(len(hue_distances)),
                key=lambda index: (rgb_distances[index], hue_distances[index]),
            )
            if hue_distances[sector_index] > sector_hue_threshold_degrees(sector_hues, sector_index):
                continue
            if rgb_distances[sector_index] > sector_rgb_threshold(sector_prototypes, sector_index):
                continue

            dx = x - site_x
            dy = y - site_y
            radius = math.sqrt((dx * dx) + (dy * dy))
            if radius < POINT_MIN_RADIUS:
                continue

            sector_masks[sector_index][y, x] = 1
    else:
        for y in range(height):
            for x in range(width):
                if not colorful_mask[y][x]:
                    continue
                if x < legend_x and y < legend_y:
                    continue

                density = neighborhood_sum(colorful_integral, x, y, POINT_WINDOW_RADIUS, width, height)
                if density < POINT_MIN_DENSITY or density > POINT_MAX_DENSITY:
                    continue

                hue, _saturation, _value = hsv_pixel(hsv_cache, x, y)
                hue_distances = [circular_hue_distance(hue, sector_hue) * 360.0 for sector_hue in sector_hues]
                pixel_rgb = rgb_pixel(bitmap, x, y)
                rgb_distances = [rgb_distance(pixel_rgb, prototype) for prototype in sector_prototypes]
                sector_index = min(
                    range(len(hue_distances)),
                    key=lambda index: (rgb_distances[index], hue_distances[index]),
                )
                if hue_distances[sector_index] > sector_hue_threshold_degrees(sector_hues, sector_index):
                    continue
                if rgb_distances[sector_index] > sector_rgb_threshold(sector_prototypes, sector_index):
                    continue

                dx = x - site_x
                dy = y - site_y
                radius = math.sqrt((dx * dx) + (dy * dy))
                if radius < POINT_MIN_RADIUS:
                    continue

                sector_masks[sector_index][y][x] = 1

    for sector_index, sector_mask in enumerate(sector_masks):
        components = extract_components(sector_mask)
        point_like_components = [component for component in components if is_point_like_component(component)]
        angles, rgb_samples = collect_component_samples(point_like_components, bitmap, site_center)
        site_angle = float(sector_signatures[sector_index]["site_angle"])
        use_site_attached_fallback = len(angles) < MIN_POINT_PIXELS_PER_COLOR
        if (
            not use_site_attached_fallback
            and angular_distance_degrees(circular_mean_degrees(angles), site_angle) > SITE_ATTACHED_FALLBACK_MISMATCH_DEG
            and point_component_near_site_ratio(point_like_components, site_center) <= POINT_COMPONENT_NEAR_SITE_RATIO_MAX
            and has_site_attached_branch_signal(components, site_center, site_angle)
        ):
            use_site_attached_fallback = True

        if use_site_attached_fallback:
            fallback_components = select_site_attached_components(
                components=components,
                site_center=site_center,
                site_angle=site_angle,
            )
            angles, rgb_samples = collect_component_samples(fallback_components, bitmap, site_center)

        if len(angles) < MIN_POINT_PIXELS_PER_COLOR:
            raise SsvAnalysisError(
                "The extracted image does not contain enough separated serving points for each sector.",
                code="insufficient_sector_points",
            )
        point_sets[sector_index]["angles"] = angles
        point_sets[sector_index]["rgb_samples"] = rgb_samples
        point_sets[sector_index]["rgb"] = average_rgb(rgb_samples)
        evaluation_angles.append(
            site_angle
            if use_site_attached_fallback
            else circular_mean_degrees(angles)
        )

    return point_sets, evaluation_angles


def extract_sector_signatures(
    bitmap: Bitmap,
    hsv_cache: list[list[tuple[float, float, float] | None]],
    colorful_mask: list[list[int]],
    colorful_integral: list[list[int]],
    sector_hues: list[float],
    site_center: tuple[float, float],
) -> list[dict[str, object]]:
    site_x, site_y = site_center
    signatures: list[dict[str, object]] = []
    scale = dense_image_scale(bitmap)
    fan_inner = scaled_radius(SITE_FAN_INNER_RADIUS, scale, SITE_FAN_INNER_RADIUS)
    fan_outer = scaled_radius(SITE_FAN_OUTER_RADIUS, scale, SITE_FAN_OUTER_RADIUS)
    fan_density_radius = scaled_radius(SITE_FAN_DENSITY_RADIUS, scale, SITE_FAN_DENSITY_RADIUS)

    for sector_hue in sector_hues:
        samples: list[tuple[int, int, int]] = []
        sample_angles: list[float] = []
        for y in range(max(0, int(site_y - fan_outer)), min(bitmap.height, int(site_y + fan_outer + 1))):
            for x in range(max(0, int(site_x - fan_outer)), min(bitmap.width, int(site_x + fan_outer + 1))):
                if not colorful_mask[y][x]:
                    continue

                dx = x - site_x
                dy = y - site_y
                radius = math.sqrt((dx * dx) + (dy * dy))
                if radius < fan_inner or radius > max(fan_inner + 1, fan_outer - 6):
                    continue

                density = neighborhood_sum(colorful_integral, x, y, fan_density_radius, bitmap.width, bitmap.height)
                if density < SITE_FAN_MIN_DENSITY:
                    continue

                hue = hsv_pixel(hsv_cache, x, y)[0]
                if circular_hue_distance(hue, sector_hue) * 360.0 > SECTOR_PROTOTYPE_HUE_WINDOW_DEG:
                    continue

                samples.append(rgb_pixel(bitmap, x, y))
                sample_angles.append(angle_from_center(dx, dy))

        if not samples:
            raise SsvAnalysisError(
                "The serving-sector colors could not be sampled reliably from the site fan.",
                code="insufficient_sector_samples",
            )

        signatures.append(
            {
                "rgb": average_rgb(samples),
                "site_angle": circular_mean_degrees(sample_angles),
            }
        )

    return signatures


def extract_components(mask: Any) -> list[dict[str, object]]:
    if np is not None and isinstance(mask, np.ndarray):
        return extract_binary_components(mask_array=mask)
    return extract_binary_components(mask_rows=mask)


def is_point_like_component(component: dict[str, object]) -> bool:
    area = int(component["area"])
    width = int(component["width"])
    height = int(component["height"])
    fill_ratio = area / max(width * height, 1)

    if area < POINT_COMPONENT_MIN_PIXELS or area > POINT_COMPONENT_MAX_PIXELS:
        return False
    if width > POINT_COMPONENT_MAX_SPAN or height > POINT_COMPONENT_MAX_SPAN:
        return False
    if fill_ratio < POINT_COMPONENT_MIN_FILL:
        return False
    return True


def is_branch_like_component(component: dict[str, object]) -> bool:
    area = int(component["area"])
    width = int(component["width"])
    height = int(component["height"])
    fill_ratio = area / max(width * height, 1)

    if area < BRANCH_COMPONENT_MIN_PIXELS:
        return False
    if max(width, height) < BRANCH_COMPONENT_MIN_SPAN:
        return False
    if fill_ratio > BRANCH_COMPONENT_MAX_FILL:
        return False
    return True


def collect_component_samples(
    components: list[dict[str, object]],
    bitmap: Bitmap,
    site_center: tuple[float, float],
) -> tuple[list[float], list[tuple[int, int, int]]]:
    site_x, site_y = site_center
    angles: list[float] = []
    rgb_samples: list[tuple[int, int, int]] = []

    for component in components:
        for x, y in component["pixels"]:
            angles.append(angle_from_center(x - site_x, y - site_y))
            rgb_samples.append(rgb_pixel(bitmap, x, y))

    return angles, rgb_samples


def select_site_attached_components(
    components: list[dict[str, object]],
    site_center: tuple[float, float],
    site_angle: float,
) -> list[dict[str, object]]:
    selected_components: list[dict[str, object]] = []

    for component in components:
        component_angles = component_angle_samples(component, site_center)
        if not component_angles:
            continue

        mean_angle = circular_mean_degrees(component_angles)
        min_distance = component_min_distance(component, site_center)
        angle_delta = angular_distance_degrees(mean_angle, site_angle)

        if is_point_like_component(component):
            if angle_delta <= SITE_ATTACHED_POINT_ANGLE_WINDOW_DEG:
                selected_components.append(component)
            elif (
                min_distance <= SITE_ATTACHED_COMPONENT_MAX_DISTANCE
                and int(component["area"]) <= SITE_ATTACHED_NEAR_SITE_POINT_MAX_PIXELS
            ):
                selected_components.append(component)
            continue

        if not is_branch_like_component(component):
            continue
        if min_distance > SITE_ATTACHED_COMPONENT_MAX_DISTANCE:
            continue
        if angle_delta > SITE_ATTACHED_BRANCH_ANGLE_WINDOW_DEG:
            continue
        selected_components.append(component)

    return selected_components


def has_site_attached_branch_signal(
    components: list[dict[str, object]],
    site_center: tuple[float, float],
    site_angle: float,
) -> bool:
    for component in components:
        if not is_branch_like_component(component):
            continue
        min_distance = component_min_distance(component, site_center)
        if min_distance > SITE_ATTACHED_COMPONENT_MAX_DISTANCE:
            continue

        component_angles = component_angle_samples(component, site_center)
        if not component_angles:
            continue
        mean_angle = circular_mean_degrees(component_angles)
        if angular_distance_degrees(mean_angle, site_angle) <= SITE_ATTACHED_BRANCH_ANGLE_WINDOW_DEG:
            return True

    return False


def component_angle_samples(component: dict[str, object], site_center: tuple[float, float]) -> list[float]:
    site_x, site_y = site_center
    return [angle_from_center(x - site_x, y - site_y) for x, y in component["pixels"]]


def component_min_distance(component: dict[str, object], site_center: tuple[float, float]) -> float:
    site_x, site_y = site_center
    return min(math.sqrt(((x - site_x) ** 2) + ((y - site_y) ** 2)) for x, y in component["pixels"])


def point_component_near_site_ratio(
    components: list[dict[str, object]],
    site_center: tuple[float, float],
) -> float:
    total_pixels = sum(int(component["area"]) for component in components)
    if total_pixels <= 0:
        return 0.0

    near_site_pixels = sum(
        int(component["area"])
        for component in components
        if component_min_distance(component, site_center) <= POINT_COMPONENT_NEAR_SITE_RADIUS
    )
    return near_site_pixels / total_pixels


def compute_misassigned_ratio(dominant_angles: list[float], point_sets: list[dict[str, object]]) -> float:
    total = 0
    misassigned = 0
    for index, point_set in enumerate(point_sets):
        for angle in point_set["angles"]:
            total += 1
            own_distance = angular_distance_degrees(angle, dominant_angles[index])
            other_distance = min(
                angular_distance_degrees(angle, dominant_angles[other_index])
                for other_index in range(len(dominant_angles))
                if other_index != index
            )
            if other_distance + MISASSIGN_MARGIN_DEG < own_distance:
                misassigned += 1
    return misassigned / total if total else 1.0


def compute_mixed_bin_ratio(point_sets: list[dict[str, object]]) -> float:
    bins = [[0, 0, 0] for _ in range(36)]
    for sector_index, point_set in enumerate(point_sets):
        for angle in point_set["angles"]:
            bins[int(angle // 10) % 36][sector_index] += 1

    mixed_pixels = 0
    total_pixels = 0
    for bucket in bins:
        bucket_total = sum(bucket)
        if bucket_total == 0:
            continue

        total_pixels += bucket_total
        populated_colors = [count for count in bucket if count > 0]
        dominant_count = max(bucket)
        if len(populated_colors) >= 2 and (dominant_count / bucket_total) < 0.85:
            mixed_pixels += bucket_total

    return mixed_pixels / total_pixels if total_pixels else 1.0


def compute_sector_boundaries(ordered_angles: list[float]) -> list[float]:
    boundaries: list[float] = []
    for index in range(len(ordered_angles)):
        left = ordered_angles[index]
        right = ordered_angles[(index + 1) % len(ordered_angles)]
        if index == len(ordered_angles) - 1 and right < left:
            right += 360.0
        boundaries.append(((left + right) / 2.0) % 360.0)
    return boundaries


def compute_intrusion_ratios(
    ordered_angles: list[float],
    boundaries: list[float],
    point_sets: list[dict[str, object]],
) -> list[float]:
    ordered_point_sets = point_sets
    intrusion_ratios: list[float] = []

    for index, point_set in enumerate(ordered_point_sets):
        start = boundaries[index - 1]
        end = boundaries[index]
        outside = 0

        for angle in point_set["angles"]:
            adjusted_angle = angle
            adjusted_start = start
            adjusted_end = end
            if adjusted_end <= adjusted_start:
                adjusted_end += 360.0
            if adjusted_angle < adjusted_start:
                adjusted_angle += 360.0
            if not (adjusted_start <= adjusted_angle < adjusted_end):
                outside += 1

        intrusion_ratios.append(outside / len(point_set["angles"]))

    return intrusion_ratios


def compute_zone_matrix(
    boundaries: list[float],
    point_sets: list[dict[str, object]],
) -> list[list[int]]:
    zone_matrix = [[0] * len(point_sets) for _ in point_sets]

    for source_index, point_set in enumerate(point_sets):
        for angle in point_set["angles"]:
            zone_matrix[source_index][zone_index_for_angle(angle, boundaries)] += 1

    return zone_matrix


def zone_index_for_angle(angle: float, boundaries: list[float]) -> int:
    for index in range(len(boundaries)):
        start = boundaries[index - 1]
        end = boundaries[index]
        adjusted_angle = angle
        adjusted_start = start
        adjusted_end = end
        if adjusted_end <= adjusted_start:
            adjusted_end += 360.0
        if adjusted_angle < adjusted_start:
            adjusted_angle += 360.0
        if adjusted_start <= adjusted_angle < adjusted_end:
            return index
    return 0


def detect_late_ho_warnings(
    detected_colors: list[DetectedColor],
    point_sets: list[dict[str, object]],
    zone_matrix: list[list[int]],
) -> list[dict[str, object]]:
    if not zone_matrix:
        return []

    row_totals = [sum(row) for row in zone_matrix]
    if not row_totals or max(row_totals) <= 0:
        return []

    max_points = max(row_totals)
    best_candidate = None

    for source_index, total in enumerate(row_totals):
        if total <= 0:
            continue

        own_ratio = zone_matrix[source_index][source_index] / total
        if own_ratio > LATE_HO_OWN_ZONE_MAX:
            continue
        if total > (max_points * LATE_HO_SOURCE_COUNT_RATIO_MAX):
            continue

        for target_index in range(len(zone_matrix)):
            if target_index == source_index:
                continue

            source_to_target_ratio = zone_matrix[source_index][target_index] / total
            if source_to_target_ratio < LATE_HO_SOURCE_TO_TARGET_MIN:
                continue

            target_total = row_totals[target_index]
            if target_total <= 0:
                continue

            target_own_ratio = zone_matrix[target_index][target_index] / target_total
            if target_own_ratio < LATE_HO_TARGET_OWN_MIN:
                continue

            angle_separation = angular_distance_degrees(
                detected_colors[source_index].dominant_angle,
                detected_colors[target_index].dominant_angle,
            )
            if angle_separation < LATE_HO_TARGET_ANGLE_MIN:
                continue

            target_zone_total = sum(zone_matrix[row_index][target_index] for row_index in range(len(zone_matrix)))
            if target_zone_total <= 0:
                continue

            other_foreign = sum(
                zone_matrix[row_index][target_index]
                for row_index in range(len(zone_matrix))
                if row_index not in (source_index, target_index)
            )
            other_foreign_ratio = other_foreign / target_zone_total
            if other_foreign_ratio > LATE_HO_TARGET_OTHER_MAX:
                continue

            score = (
                (source_to_target_ratio * 0.45)
                + (target_own_ratio * 0.30)
                + (min(angle_separation / 180.0, 1.0) * 0.15)
                + ((1.0 - other_foreign_ratio) * 0.10)
            )
            candidate = (score, source_index, target_index)
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

    if best_candidate is None:
        return []

    _score, source_index, target_index = best_candidate
    source_name = describe_sector_color(detected_colors[source_index].rgb)
    target_name = describe_sector_color(detected_colors[target_index].rgb)
    return [
        {
            "kind": "late_ho",
            "source_index": source_index,
            "target_index": target_index,
            "source_color": source_name,
            "target_color": target_name,
            "message": f"Possible late HO from {source_name} sector to {target_name} sector",
        }
    ]


def detect_pair_late_ho_warnings(
    detected_colors: list[DetectedColor],
    point_sets: list[dict[str, object]],
    zone_matrix: list[list[int]],
    intrusion_ratios: list[float],
    misassigned_ratio: float,
    mixed_bin_ratio: float,
) -> list[dict[str, object]]:
    if not zone_matrix or not intrusion_ratios:
        return []
    if misassigned_ratio > PAIR_LATE_HO_MISASSIGNED_MAX:
        return []
    if mixed_bin_ratio > PAIR_LATE_HO_MIXED_MAX:
        return []

    row_totals = [sum(row) for row in zone_matrix]
    best_candidate = None

    for source_index, total in enumerate(row_totals):
        if total <= 0:
            continue

        own_ratio = zone_matrix[source_index][source_index] / total
        if own_ratio < PAIR_LATE_HO_SOURCE_OWN_MIN:
            continue

        other_indices = [index for index in range(len(zone_matrix)) if index != source_index]
        target_index = max(other_indices, key=lambda index: zone_matrix[source_index][index])
        source_to_target_ratio = zone_matrix[source_index][target_index] / total
        if source_to_target_ratio < PAIR_LATE_HO_SOURCE_TO_TARGET_MIN:
            continue
        if intrusion_ratios[source_index] < INTRUSION_THRESHOLD or intrusion_ratios[source_index] > PAIR_LATE_HO_INTRUSION_MAX:
            continue

        other_ratio = sum(
            zone_matrix[source_index][index]
            for index in range(len(zone_matrix))
            if index not in (source_index, target_index)
        ) / total
        if other_ratio > PAIR_LATE_HO_SOURCE_OTHER_MAX:
            continue

        target_total = row_totals[target_index]
        if target_total <= 0:
            continue
        target_own_ratio = zone_matrix[target_index][target_index] / target_total
        if target_own_ratio < PAIR_LATE_HO_TARGET_OWN_MIN:
            continue

        target_zone_total = sum(zone_matrix[row_index][target_index] for row_index in range(len(zone_matrix)))
        if target_zone_total <= 0:
            continue
        target_other_ratio = sum(
            zone_matrix[row_index][target_index]
            for row_index in range(len(zone_matrix))
            if row_index not in (source_index, target_index)
        ) / target_zone_total
        if target_other_ratio > PAIR_LATE_HO_TARGET_OTHER_MAX:
            continue

        angle_separation = angular_distance_degrees(
            detected_colors[source_index].dominant_angle,
            detected_colors[target_index].dominant_angle,
        )
        if angle_separation < PAIR_LATE_HO_ANGLE_MIN:
            continue

        score = (
            (source_to_target_ratio * 0.45)
            + (own_ratio * 0.20)
            + (target_own_ratio * 0.20)
            + ((1.0 - target_other_ratio) * 0.10)
            + (min(angle_separation / 180.0, 1.0) * 0.05)
        )
        candidate = (score, source_index, target_index)
        if best_candidate is None or candidate[0] > best_candidate[0]:
            best_candidate = candidate

    if best_candidate is None:
        return []

    _score, source_index, target_index = best_candidate
    source_name = describe_sector_color(detected_colors[source_index].rgb)
    target_name = describe_sector_color(detected_colors[target_index].rgb)
    return [
        {
            "kind": "late_ho_pair",
            "source_index": source_index,
            "target_index": target_index,
            "source_color": source_name,
            "target_color": target_name,
            "message": f"Possible late HO from {source_name} sector to {target_name} sector",
        },
        {
            "kind": "late_ho_pair",
            "source_index": target_index,
            "target_index": source_index,
            "source_color": target_name,
            "target_color": source_name,
            "message": f"Possible late HO from {target_name} sector to {source_name} sector",
        },
    ]


def detect_minor_pair_late_ho_warnings(
    detected_colors: list[DetectedColor],
    point_sets: list[dict[str, object]],
    zone_matrix: list[list[int]],
    intrusion_ratios: list[float],
    misassigned_ratio: float,
    mixed_bin_ratio: float,
) -> list[dict[str, object]]:
    if not zone_matrix or not intrusion_ratios:
        return []
    if misassigned_ratio > MINOR_PAIR_HO_MISASSIGNED_MAX:
        return []
    if mixed_bin_ratio > MINOR_PAIR_HO_MIXED_MAX:
        return []

    row_totals = [sum(row) for row in zone_matrix]
    best_candidate = None

    for source_index, total in enumerate(row_totals):
        if total <= 0:
            continue

        other_indices = [index for index in range(len(zone_matrix)) if index != source_index]
        target_index = max(other_indices, key=lambda index: zone_matrix[source_index][index])
        source_to_target_ratio = zone_matrix[source_index][target_index] / total
        if source_to_target_ratio < MINOR_PAIR_HO_SOURCE_TO_TARGET_MIN:
            continue
        if source_to_target_ratio > MINOR_PAIR_HO_SOURCE_TO_TARGET_MAX:
            continue
        if intrusion_ratios[source_index] > MINOR_PAIR_HO_INTRUSION_MAX:
            continue

        target_total = row_totals[target_index]
        if target_total <= 0:
            continue
        target_own_ratio = zone_matrix[target_index][target_index] / target_total
        if target_own_ratio < MINOR_PAIR_HO_TARGET_OWN_MIN:
            continue

        angle_separation = angular_distance_degrees(
            detected_colors[source_index].dominant_angle,
            detected_colors[target_index].dominant_angle,
        )
        if angle_separation < MINOR_PAIR_HO_ANGLE_MIN:
            continue

        score = (
            (source_to_target_ratio * 0.45)
            + (target_own_ratio * 0.35)
            + (min(angle_separation / 180.0, 1.0) * 0.20)
        )
        candidate = (score, source_index, target_index)
        if best_candidate is None or candidate[0] > best_candidate[0]:
            best_candidate = candidate

    if best_candidate is None:
        return []

    _score, source_index, target_index = best_candidate
    source_name = describe_sector_color(detected_colors[source_index].rgb)
    target_name = describe_sector_color(detected_colors[target_index].rgb)
    return [
        {
            "kind": "late_ho_minor_pair",
            "source_index": source_index,
            "target_index": target_index,
            "source_color": source_name,
            "target_color": target_name,
            "message": f"Possible late HO from {source_name} sector to {target_name} sector",
        },
        {
            "kind": "late_ho_minor_pair",
            "source_index": target_index,
            "target_index": source_index,
            "source_color": target_name,
            "target_color": source_name,
            "message": f"Possible late HO from {target_name} sector to {source_name} sector",
        },
    ]


def compute_min_angle_separation(angles: list[float]) -> float:
    separations = []
    for index, angle in enumerate(angles):
        for other_index in range(index + 1, len(angles)):
            separations.append(angular_distance_degrees(angle, angles[other_index]))
    return min(separations) if separations else 0.0


def compute_confidence(
    total_points: int,
    min_angle_separation: float,
    misassigned_ratio: float,
    mixed_bin_ratio: float,
    intrusion_ratio: float,
) -> float:
    point_score = min(total_points / 1800.0, 1.0)
    angle_score = min(min_angle_separation / 110.0, 1.0)
    contamination = max(
        misassigned_ratio / max(MISASSIGNED_THRESHOLD, 0.001),
        mixed_bin_ratio / max(MIXED_BIN_THRESHOLD, 0.001),
        intrusion_ratio / max(INTRUSION_THRESHOLD, 0.001),
    )
    stability_score = max(0.0, 1.0 - (contamination * 0.25))
    confidence = 0.35 + (0.35 * point_score) + (0.20 * angle_score) + (0.10 * stability_score)
    return max(0.05, min(confidence, 0.99))


def build_annotated_preview(
    bitmap: Bitmap,
    preview_image_uri: str,
    site_center: tuple[float, float],
    detected_colors: list[DetectedColor],
    sector_boundaries: list[float],
    verdict: str,
    metrics: dict[str, object],
) -> str:
    site_x, site_y = site_center
    width = bitmap.width
    height = bitmap.height
    overlay_lines: list[str] = []

    for detected_color in detected_colors:
        x2, y2 = project_angle(site_x, site_y, detected_color.dominant_angle, 80.0)
        overlay_lines.append(
            f'<line x1="{site_x:.2f}" y1="{site_y:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{detected_color.hex}" stroke-width="4" stroke-linecap="round" />'
        )

    for boundary in sector_boundaries:
        x2, y2 = project_angle(site_x, site_y, boundary, 72.0)
        overlay_lines.append(
            f'<line x1="{site_x:.2f}" y1="{site_y:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#ffffff" stroke-width="2" stroke-dasharray="5 4" opacity="0.65" />'
        )

    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="{preview_image_uri}" width="{width}" height="{height}" />
  {''.join(overlay_lines)}
  <circle cx="{site_x:.2f}" cy="{site_y:.2f}" r="8" fill="none" stroke="#ffffff" stroke-width="3" />
  <circle cx="{site_x:.2f}" cy="{site_y:.2f}" r="3" fill="#ffffff" />
</svg>
""".strip()

    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def project_angle(x: float, y: float, angle_degrees: float, distance: float) -> tuple[float, float]:
    radians = math.radians(angle_degrees)
    return (
        x + (math.cos(radians) * distance),
        y - (math.sin(radians) * distance),
    )


def build_integral_image(mask: Any) -> Any:
    return build_integral_image_shared(mask)


def neighborhood_sum(
    integral: Any,
    x: int,
    y: int,
    radius: int,
    width: int,
    height: int,
) -> int:
    return neighborhood_sum_shared(integral, x, y, radius, width, height)


def average_rgb(samples: Iterable[tuple[int, int, int]]) -> tuple[int, int, int]:
    sample_list = list(samples)
    total = len(sample_list)
    red = sum(sample[0] for sample in sample_list) // total
    green = sum(sample[1] for sample in sample_list) // total
    blue = sum(sample[2] for sample in sample_list) // total
    return red, green, blue


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def describe_sector_color(rgb: tuple[int, int, int]) -> str:
    hue, saturation, value = rgb_to_hsv(*rgb)
    hue_degrees = hue * 360.0

    if value < 0.20 or saturation < 0.20:
        return "Neutral"
    if hue_degrees < 20 or hue_degrees >= 340:
        return "Red"
    if hue_degrees < 50:
        return "Orange"
    if hue_degrees < 75:
        return "Yellow"
    if hue_degrees < 165:
        return "Green"
    if hue_degrees < 205:
        return "Cyan"
    if hue_degrees < 280:
        return "Blue"
    return "Purple"


def rgb_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def sector_rgb_threshold(sector_prototypes: list[tuple[int, int, int]], sector_index: int) -> float:
    prototype = sector_prototypes[sector_index]
    neighbor_distances = [
        rgb_distance(prototype, other)
        for other_index, other in enumerate(sector_prototypes)
        if other_index != sector_index
    ]
    if not neighbor_distances:
        return SECTOR_RGB_MATCH_CAP
    return min(SECTOR_RGB_MATCH_CAP, max(SECTOR_RGB_MATCH_FLOOR, min(neighbor_distances) * 0.42))


def sector_hue_threshold_degrees(sector_hues: list[float], sector_index: int) -> float:
    hue = sector_hues[sector_index]
    neighbor_distances = [
        circular_hue_distance(hue, other) * 360.0
        for other_index, other in enumerate(sector_hues)
        if other_index != sector_index
    ]
    if not neighbor_distances:
        return SECTOR_HUE_MATCH_CAP_DEG
    return min(SECTOR_HUE_MATCH_CAP_DEG, max(SECTOR_HUE_MATCH_FLOOR_DEG, min(neighbor_distances) * 0.42))


def circular_bin_distance(left: int, right: int) -> int:
    distance = abs(left - right) % 36
    return min(distance, 36 - distance)


def circular_hue_distance(left: float, right: float) -> float:
    distance = abs(left - right) % 1.0
    return min(distance, 1.0 - distance)


def rgb_to_hsv(red: int, green: int, blue: int) -> tuple[float, float, float]:
    red_f = red / 255.0
    green_f = green / 255.0
    blue_f = blue / 255.0
    value = max(red_f, green_f, blue_f)
    min_value = min(red_f, green_f, blue_f)
    delta = value - min_value

    saturation = 0.0 if value == 0 else delta / value

    if delta == 0:
        hue = 0.0
    elif value == red_f:
        hue = ((green_f - blue_f) / delta) % 6.0
    elif value == green_f:
        hue = ((blue_f - red_f) / delta) + 2.0
    else:
        hue = ((red_f - green_f) / delta) + 4.0

    return (hue / 6.0) % 1.0, saturation, value


def angle_from_center(dx: float, dy: float) -> float:
    return (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0


def circular_mean_degrees(angles: list[float]) -> float:
    sin_sum = sum(math.sin(math.radians(angle)) for angle in angles)
    cos_sum = sum(math.cos(math.radians(angle)) for angle in angles)
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0


def angular_distance_degrees(left: float, right: float) -> float:
    distance = abs(left - right) % 360.0
    return min(distance, 360.0 - distance)
