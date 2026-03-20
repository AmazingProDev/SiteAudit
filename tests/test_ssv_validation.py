from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from ssv_validation.analyzer import (
    analyze_bitmap,
    detect_late_ho_warnings,
    detect_minor_pair_late_ho_warnings,
    detect_pair_late_ho_warnings,
)
from ssv_validation.kpi_analyzer import analyze_kpi_bitmap
from ssv_validation.legend_mapping import (
    azimuth_confirmation_bonus,
    extract_identifier_lookup_from_sheet,
    row_identifier_cost,
)
from ssv_validation.models import AnalysisOutcome, Bitmap, DetectedColor, ImageCandidate
from ssv_validation.service import validate_ssv_workbook
from ssv_validation.throughput import evaluate_avg_throughput
from ssv_validation.workbook import infer_lte_band, match_target_profile, rank_sheet_name


WHITE = (247, 247, 247)
ROAD = (192, 202, 214)
BLUE = (37, 72, 235)
YELLOW = (240, 224, 40)
RED = (234, 52, 45)


def make_canvas(width: int = 583, height: int = 286) -> list[list[tuple[int, int, int]]]:
    pixels = [[WHITE for _ in range(width)] for _ in range(height)]

    for x in range(0, width, 58):
        for y in range(height):
            pixels[y][x] = ROAD
            if x + 1 < width:
                pixels[y][x + 1] = ROAD

    for y in range(0, height, 48):
        for x in range(width):
            pixels[y][x] = ROAD
            if y + 1 < height:
                pixels[y + 1][x] = ROAD

    add_legend(pixels)
    return pixels


def add_legend(pixels: list[list[tuple[int, int, int]]]) -> None:
    legend_colors = [BLUE, YELLOW, RED]
    for index, color in enumerate(legend_colors):
        y_start = 8 + (index * 16)
        for y in range(y_start, y_start + 10):
            for x in range(8, 18):
                pixels[y][x] = color


def draw_disc(pixels: list[list[tuple[int, int, int]]], cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    width = len(pixels[0])
    height = len(pixels)
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if ((x - cx) ** 2) + ((y - cy) ** 2) <= (radius * radius):
                pixels[y][x] = color


def draw_branch(
    pixels: list[list[tuple[int, int, int]]],
    site: tuple[int, int],
    angle_degrees: float,
    length: int,
    color: tuple[int, int, int],
    point_step: int = 10,
) -> None:
    site_x, site_y = site
    radians = math.radians(angle_degrees)

    for radius in range(12, 38, 3):
        x = int(round(site_x + (radius * math.cos(radians))))
        y = int(round(site_y - (radius * math.sin(radians))))
        draw_disc(pixels, x, y, 4, color)

    for radius in range(40, length, point_step):
        x = int(round(site_x + (radius * math.cos(radians))))
        y = int(round(site_y - (radius * math.sin(radians))))
        draw_disc(pixels, x, y, 3, color)


def draw_connected_branch(
    pixels: list[list[tuple[int, int, int]]],
    site: tuple[int, int],
    angle_degrees: float,
    length: int,
    color: tuple[int, int, int],
    step: int = 2,
) -> None:
    site_x, site_y = site
    radians = math.radians(angle_degrees)

    for radius in range(12, length, step):
        x = int(round(site_x + (radius * math.cos(radians))))
        y = int(round(site_y - (radius * math.sin(radians))))
        draw_disc(pixels, x, y, 2, color)


def make_no_cross_bitmap() -> Bitmap:
    pixels = make_canvas()
    site = (360, 88)
    draw_branch(pixels, site, 90.0, 95, BLUE)
    draw_branch(pixels, site, 182.0, 180, YELLOW)
    draw_branch(pixels, site, 320.0, 185, RED)
    return Bitmap(width=len(pixels[0]), height=len(pixels), pixels=pixels)


def make_cross_bitmap() -> Bitmap:
    pixels = make_canvas()
    site = (360, 88)
    draw_branch(pixels, site, 90.0, 95, BLUE)
    draw_branch(pixels, site, 182.0, 180, YELLOW)
    draw_branch(pixels, site, 320.0, 185, RED)

    for extra_angle in range(145, 205, 8):
        draw_branch(pixels, site, float(extra_angle), 110, BLUE, point_step=18)

    for extra_angle in range(280, 345, 8):
        draw_branch(pixels, site, float(extra_angle), 115, YELLOW, point_step=18)

    return Bitmap(width=len(pixels[0]), height=len(pixels), pixels=pixels)


def make_gsm_trace_bitmap() -> Bitmap:
    pixels = make_canvas()
    site = (300, 134)
    draw_connected_branch(pixels, site, 112.0, 112, (12, 252, 252))
    draw_connected_branch(pixels, site, 285.0, 150, (9, 9, 250))
    draw_branch(pixels, site, 190.0, 150, (251, 77, 11), point_step=8)

    # Orange/red map UI noise that should not be mistaken for the serving branch.
    for cx, cy in ((455, 66), (490, 94), (380, 228), (530, 82)):
        draw_disc(pixels, cx, cy, 4, (251, 77, 11))

    return Bitmap(width=len(pixels[0]), height=len(pixels), pixels=pixels)


def make_kpi_bitmap(with_red_cluster: bool) -> Bitmap:
    pixels = make_canvas(520, 250)
    colors = [
        (36, 205, 72),
        (243, 217, 41),
        (49, 93, 247),
    ]

    for index, color in enumerate(colors):
        for step in range(12):
            draw_disc(pixels, 90 + (step * 18), 120 + (index * 18), 3, color)

    if with_red_cluster:
        for step in range(7):
            draw_disc(pixels, 250 + (step * 10), 148, 3, RED)
        for step in range(3):
            draw_disc(pixels, 130 + (step * 12), 88, 3, RED)
    else:
        for step in range(2):
            draw_disc(pixels, 250 + (step * 12), 148, 3, RED)

    return Bitmap(width=len(pixels[0]), height=len(pixels), pixels=pixels)


class WorkbookSelectionTests(unittest.TestCase):
    def test_mobility_sheet_ranks_above_other_sheets(self) -> None:
        target_score = rank_sheet_name("3. L800 DT en mobilite")
        cover_score = rank_sheet_name("Cover")
        volte_score = rank_sheet_name("5. L800 Volte en Mobilite")

        self.assertGreater(target_score, cover_score)
        self.assertGreater(target_score, volte_score)

    def test_target_profiles_detect_serving_cell_id_and_best_server(self) -> None:
        serving_match = match_target_profile("Cross Check: Serving Cell ID")
        best_server_match = match_target_profile("Installation Check: Best Server")
        rxlev_match = match_target_profile("CS RxLev")
        sinr_match = match_target_profile("800 Cells Qualité (SINR)")
        rscp_match = match_target_profile("› Best RSCP in Active set in connect state")
        ecio_match = match_target_profile("› Best Ec/Io in Active set in connect state")
        throughput_dl_match = match_target_profile("800 Cells Débit DL en mobilité (RLC)")
        throughput_ul_match = match_target_profile("800 Cells Débit UL en mobilité (RLC)")

        self.assertIsNotNone(serving_match)
        self.assertEqual(serving_match["key"], "serving_cell_id")
        self.assertIsNotNone(best_server_match)
        self.assertEqual(best_server_match["key"], "best_server")
        self.assertIsNotNone(rxlev_match)
        self.assertEqual(rxlev_match["analysis_kind"], "degradation")
        self.assertEqual(rxlev_match["metric_name"], "RxLev")
        self.assertIsNotNone(sinr_match)
        self.assertEqual(sinr_match["key"], "quality_sinr")
        self.assertIsNotNone(rscp_match)
        self.assertEqual(rscp_match["key"], "coverage_rscp")
        self.assertIsNotNone(ecio_match)
        self.assertEqual(ecio_match["key"], "quality_ecno")
        self.assertIsNotNone(throughput_dl_match)
        self.assertEqual(throughput_dl_match["key"], "throughput_dl")
        self.assertIsNotNone(throughput_ul_match)
        self.assertEqual(throughput_ul_match["key"], "throughput_ul")

    def test_infer_lte_band_detects_supported_bands(self) -> None:
        self.assertEqual(infer_lte_band("3. L800 DT en mobilite"), "L800")
        self.assertEqual(infer_lte_band("2. L2100 DT en mobilite"), "L2100")
        self.assertEqual(infer_lte_band("1. L2600 DT en mobilite"), "L2600")


class AnalysisTests(unittest.TestCase):
    def test_no_cross_bitmap_returns_no_cross(self) -> None:
        result = analyze_bitmap(make_no_cross_bitmap(), "data:image/png;base64,")
        self.assertFalse(result.cross)
        self.assertEqual(result.verdict, "No cross detected")
        self.assertEqual(len(result.detected_colors), 3)

    def test_cross_bitmap_returns_cross(self) -> None:
        result = analyze_bitmap(make_cross_bitmap(), "data:image/png;base64,")
        self.assertTrue(result.cross)
        self.assertEqual(result.verdict, "Cross detected")
        self.assertGreater(result.metrics["misassigned_pixel_ratio"], 0.18)

    def test_gsm_trace_bitmap_uses_site_attached_fallback(self) -> None:
        result = analyze_bitmap(make_gsm_trace_bitmap(), "data:image/png;base64,")

        self.assertEqual(len(result.detected_colors), 3)
        self.assertGreater(result.metrics["total_point_pixels"], 300)

    def test_kpi_bitmap_returns_ssv_nok_for_continuous_red_cluster(self) -> None:
        result = analyze_kpi_bitmap(make_kpi_bitmap(with_red_cluster=True), "data:image/png;base64,", "RSRP", "coverage")

        self.assertTrue(result.is_failure)
        self.assertEqual(result.verdict, "SSV NOK")
        self.assertIn("Continuous red points detected", " ".join(result.warnings))

    def test_kpi_bitmap_returns_ssv_ok_for_low_red_ratio(self) -> None:
        result = analyze_kpi_bitmap(make_kpi_bitmap(with_red_cluster=False), "data:image/png;base64,", "RSRP", "coverage")

        self.assertFalse(result.is_failure)
        self.assertEqual(result.verdict, "SSV OK")

    def test_throughput_map_ignores_scattered_red_ratio_without_cluster(self) -> None:
        result = analyze_kpi_bitmap(make_kpi_bitmap(with_red_cluster=False), "data:image/png;base64,", "DL Throughput", "throughput")

        self.assertFalse(result.is_failure)
        self.assertEqual(result.verdict, "SSV OK")

    def test_late_ho_warning_downgrades_one_way_intrusion(self) -> None:
        detected_colors = [
            DetectedColor("sector_1", BLUE, "#2548eb", 102.13, 236),
            DetectedColor("sector_2", RED, "#ea342d", 285.68, 182),
            DetectedColor("sector_3", YELLOW, "#f0e028", 308.04, 915),
        ]
        point_sets = [
            {"angles": [0.0] * 236},
            {"angles": [0.0] * 182},
            {"angles": [0.0] * 915},
        ]
        zone_matrix = [
            [221, 0, 15],
            [64, 0, 118],
            [113, 369, 433],
        ]

        warnings = detect_late_ho_warnings(detected_colors, point_sets, zone_matrix)

        self.assertEqual(
            warnings,
            [
                {
                    "kind": "late_ho",
                    "source_index": 1,
                    "target_index": 0,
                    "source_color": "Red",
                    "target_color": "Blue",
                    "message": "Possible late HO from Red sector to Blue sector",
                }
            ],
        )

    def test_pair_late_ho_warning_downgrades_localized_two_sector_overlap(self) -> None:
        detected_colors = [
            DetectedColor("sector_1", (8, 254, 254), "#08fefe", 66.92, 622),
            DetectedColor("sector_2", (253, 85, 14), "#fd550e", 189.07, 338),
            DetectedColor("sector_3", BLUE, "#0404fd", 316.66, 783),
        ]
        point_sets = [
            {"angles": [0.0] * 622},
            {"angles": [0.0] * 338},
            {"angles": [0.0] * 783},
        ]
        zone_matrix = [
            [622, 0, 0],
            [3, 273, 62],
            [14, 0, 769],
        ]
        intrusion_ratios = [0.0, 62 / 338, 14 / 783]

        warnings = detect_pair_late_ho_warnings(
            detected_colors=detected_colors,
            point_sets=point_sets,
            zone_matrix=zone_matrix,
            intrusion_ratios=intrusion_ratios,
            misassigned_ratio=0.0275,
            mixed_bin_ratio=0.1836,
        )

        self.assertEqual(
            warnings,
            [
                {
                    "kind": "late_ho_pair",
                    "source_index": 1,
                    "target_index": 2,
                    "source_color": "Red",
                    "target_color": "Blue",
                    "message": "Possible late HO from Red sector to Blue sector",
                },
                {
                    "kind": "late_ho_pair",
                    "source_index": 2,
                    "target_index": 1,
                    "source_color": "Blue",
                    "target_color": "Red",
                    "message": "Possible late HO from Blue sector to Red sector",
                },
            ],
        )

    def test_minor_pair_late_ho_warning_catches_small_adjacent_boundary_trace(self) -> None:
        detected_colors = [
            DetectedColor("sector_1", (8, 254, 254), "#08fefe", 66.92, 622),
            DetectedColor("sector_2", (253, 85, 14), "#fd550e", 189.07, 271),
            DetectedColor("sector_3", BLUE, "#0404fd", 316.66, 783),
        ]
        point_sets = [
            {"angles": [0.0] * 622},
            {"angles": [0.0] * 271},
            {"angles": [0.0] * 783},
        ]
        zone_matrix = [
            [622, 0, 0],
            [3, 268, 0],
            [14, 0, 769],
        ]
        intrusion_ratios = [0.0, 3 / 271, 14 / 783]

        warnings = detect_minor_pair_late_ho_warnings(
            detected_colors=detected_colors,
            point_sets=point_sets,
            zone_matrix=zone_matrix,
            intrusion_ratios=intrusion_ratios,
            misassigned_ratio=0.0042,
            mixed_bin_ratio=0.0,
        )

        self.assertEqual(
            warnings,
            [
                {
                    "kind": "late_ho_minor_pair",
                    "source_index": 2,
                    "target_index": 0,
                    "source_color": "Blue",
                    "target_color": "Cyan",
                    "message": "Possible late HO from Blue sector to Cyan sector",
                },
                {
                    "kind": "late_ho_minor_pair",
                    "source_index": 0,
                    "target_index": 2,
                    "source_color": "Cyan",
                    "target_color": "Blue",
                    "message": "Possible late HO from Cyan sector to Blue sector",
                },
            ],
        )


class ServiceTests(unittest.TestCase):
    def test_validate_workbook_keeps_multiple_same_target_images(self) -> None:
        candidates = [
            (
                ImageCandidate(
                    sheet_name="3. L800 DT en mobilite",
                    sheet_path="xl/worksheets/sheet3.xml",
                    drawing_path="xl/drawings/drawing3.xml",
                    media_path="xl/media/image5.png",
                    target_key="serving_pci",
                    target_label="Serving PCI",
                    anchor_row=2,
                    anchor_col=9,
                    score=820.0,
                    nearby_text=["J2: 800 Cells PCI de la cellules serveuse"],
                    caption_ref="J2",
                ),
                b"\x89PNG\r\n\x1a\nmock-1",
                "image/png",
            ),
            (
                ImageCandidate(
                    sheet_name="5. L800 Volte en Mobilite",
                    sheet_path="xl/worksheets/sheet5.xml",
                    drawing_path="xl/drawings/drawing5.xml",
                    media_path="xl/media/image13.png",
                    target_key="serving_pci",
                    target_label="Serving PCI",
                    anchor_row=18,
                    anchor_col=0,
                    score=694.0,
                    nearby_text=["A18: 800 Cells PCI de la cellules serveuse"],
                    caption_ref="A18",
                ),
                b"\x89PNG\r\n\x1a\nmock-2",
                "image/png",
            ),
        ]
        outcomes = [
            AnalysisOutcome(
                cross=False,
                verdict="No cross detected",
                detected_colors=[
                    DetectedColor("sector_1", BLUE, "#2548eb", 90.0, 12),
                    DetectedColor("sector_2", YELLOW, "#f0e028", 182.0, 18),
                    DetectedColor("sector_3", RED, "#ea342d", 320.0, 16),
                ],
                metrics={"confidence": 0.91},
                site_center={"x": 360.0, "y": 88.0},
                annotated_preview="data:image/svg+xml;base64,one",
            ),
            AnalysisOutcome(
                cross=True,
                verdict="Cross detected",
                detected_colors=[
                    DetectedColor("sector_1", BLUE, "#2548eb", 90.0, 12),
                    DetectedColor("sector_2", YELLOW, "#f0e028", 182.0, 18),
                    DetectedColor("sector_3", RED, "#ea342d", 320.0, 16),
                ],
                metrics={"confidence": 0.73},
                site_center={"x": 360.0, "y": 88.0},
                annotated_preview="data:image/svg+xml;base64,two",
            ),
        ]

        with (
            patch("ssv_validation.service.select_target_images", return_value=candidates),
            patch("ssv_validation.service.convert_image_to_bmp"),
            patch(
                "ssv_validation.service.decode_bmp",
                side_effect=[Bitmap(width=1, height=1, pixels=[[(0, 0, 0)]]), Bitmap(width=1, height=1, pixels=[[(0, 0, 0)]])],
            ),
            patch("ssv_validation.service.analyze_bitmap", side_effect=outcomes),
            patch("ssv_validation.service.extract_avg_throughput_metrics", return_value=None),
        ):
            result = validate_ssv_workbook(b"fake", "fake.xlsx")

        self.assertEqual(result["analysisCount"], 2)
        self.assertEqual(result["verdict"], "Cross detected")
        self.assertEqual(
            [analysis["selection"]["sheetName"] for analysis in result["analyses"]],
            ["3. L800 DT en mobilite", "5. L800 Volte en Mobilite"],
        )

    def test_validate_workbook_marks_ssv_nok_when_avg_throughput_is_below_threshold(self) -> None:
        candidates = [
            (
                ImageCandidate(
                    sheet_name="3. L800 DT en mobilite",
                    sheet_path="xl/worksheets/sheet3.xml",
                    drawing_path="xl/drawings/drawing3.xml",
                    media_path="xl/media/image8.png",
                    target_key="throughput_dl",
                    target_label="Débit DL",
                    anchor_row=34,
                    anchor_col=0,
                    score=810.0,
                    analysis_kind="degradation",
                    metric_group="throughput",
                    metric_name="DL Throughput",
                    nearby_text=["A34: 800 Cells Débit DL en mobilité (RLC)"],
                    caption_ref="A34",
                ),
                b"\x89PNG\r\n\x1a\nmock-1",
                "image/png",
            )
        ]
        throughput_average_input = {
            "sheet_name": "3. L800 DT en mobilite",
            "band": "L800",
            "dl_value_mbps": 12.0,
            "ul_value_mbps": 8.0,
            "dl_threshold_mbps": 15.0,
            "ul_threshold_mbps": 10.0,
            "dl_label_ref": "T46",
            "ul_label_ref": "T45",
            "dl_value_ref": "U46",
            "ul_value_ref": "U45",
        }

        with (
            patch("ssv_validation.service.select_target_images", return_value=candidates),
            patch("ssv_validation.service.convert_image_to_bmp"),
            patch("ssv_validation.service.decode_bmp", return_value=Bitmap(width=1, height=1, pixels=[[(0, 0, 0)]])),
            patch(
                "ssv_validation.service.analyze_kpi_bitmap",
                return_value=AnalysisOutcome(
                    cross=False,
                    verdict="SSV OK",
                    detected_colors=[],
                    metrics={"metric_name": "DL Throughput", "metric_group": "throughput"},
                    site_center={"x": 0.0, "y": 0.0},
                    annotated_preview="data:image/svg+xml;base64,map",
                    analysis_kind="degradation",
                    is_failure=False,
                ),
            ),
            patch("ssv_validation.service.extract_avg_throughput_metrics", return_value=throughput_average_input),
        ):
            result = validate_ssv_workbook(b"fake", "fake.xlsx")

        self.assertEqual(result["verdict"], "SSV NOK")
        self.assertTrue(result["isFailure"])
        self.assertEqual(result["analyses"][-1]["label"], "Avg Throughput")
        self.assertEqual(result["analyses"][-1]["verdict"], "SSV NOK")


class LegendMappingTests(unittest.TestCase):
    def test_extract_identifier_lookup_from_sheet_reads_cell_name_table(self) -> None:
        cells = {
            (3, 19): "Cell Name",
            (3, 20): "Cell ID",
            (4, 19): "2G_CAS_CasaAkidAllamDep_20764",
            (4, 20): "23435",
            (5, 19): "2G_CAS_CasaAkidAllamDep_20765",
            (5, 20): "23436",
            (6, 19): "2G_CAS_CasaAkidAllamDep_20766",
            (6, 20): "23440",
        }

        lookup = extract_identifier_lookup_from_sheet(cells, ("cell id",))

        self.assertEqual(
            lookup,
            {
                "23435": {"name": "2G_CAS_CasaAkidAllamDep_20764", "azimuth": None},
                "23436": {"name": "2G_CAS_CasaAkidAllamDep_20765", "azimuth": None},
                "23440": {"name": "2G_CAS_CasaAkidAllamDep_20766", "azimuth": None},
            },
        )

    def test_row_identifier_cost_handles_ocr_digit_error(self) -> None:
        self.assertLess(row_identifier_cost("23430 (200,37.88%)", "23436"), 2)

    def test_azimuth_confirmation_bonus_rewards_matching_site_angle(self) -> None:
        detected_color = DetectedColor("sector_1", BLUE, "#2548eb", 100.0, 100, site_angle=92.0)
        bonus = azimuth_confirmation_bonus(detected_color, {"name": "cell", "azimuth": 0.0})

        self.assertGreater(bonus, 20.0)


class ThroughputTests(unittest.TestCase):
    def test_evaluate_avg_throughput_uses_band_thresholds(self) -> None:
        result = evaluate_avg_throughput(
            {
                "band": "L2100",
                "dl_value_mbps": 24.0,
                "ul_value_mbps": 16.0,
            }
        )

        self.assertTrue(result["isFailure"])
        self.assertEqual(result["verdict"], "SSV NOK")
        self.assertIn("Avg DL Throughput 24.00 Mbps is below the L2100 minimum of 25 Mbps.", result["warnings"])


if __name__ == "__main__":
    unittest.main()
