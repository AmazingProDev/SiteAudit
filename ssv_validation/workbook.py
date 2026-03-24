from __future__ import annotations

import io
import logging
import mimetypes
import posixpath
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from difflib import SequenceMatcher

from .models import ImageCandidate, WorkbookSheet

LOGGER = logging.getLogger(__name__)

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
DRAW_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

TARGET_SHEET_HINTS = (
    "3 l800 dt en mobilite",
    "l800 dt en mobilite",
    "lte 800 mobility",
    "800 mobility",
)
TARGET_IMAGE_PROFILES = (
    {
        "key": "serving_pci",
        "label": "Serving PCI",
        "analysis_kind": "cross",
        "phrases": (
            "800 cells pci de la cellules serveuse",
            "2600 cells pci de la cellules serveuse",
            "1800 cells pci de la cellules serveuse",
            "cells pci de la cellules serveuse",
            "pci de la cellules serveuse",
            "serving cell pci",
            "serving pci",
        ),
        "tokens": ("cells", "pci", "serveuse"),
    },
    {
        "key": "serving_cell_id",
        "label": "Serving Cell ID",
        "analysis_kind": "cross",
        "phrases": ("serving cell id",),
        "tokens": ("serving", "cell", "id"),
    },
    {
        "key": "best_server",
        "label": "Best Server",
        "analysis_kind": "cross",
        "phrases": ("best server",),
        "tokens": ("best", "server"),
    },
    {
        "key": "cs_rxlev",
        "label": "CS RxLev",
        "analysis_kind": "degradation",
        "metric_group": "coverage",
        "metric_name": "RxLev",
        "phrases": ("cs rxlev",),
        "tokens": ("rxlev",),
    },
    {
        "key": "cs_rxqual",
        "label": "CS RxQual",
        "analysis_kind": "degradation",
        "metric_group": "quality",
        "metric_name": "RxQual",
        "phrases": ("cs rxqual",),
        "tokens": ("rxqual",),
    },
    {
        "key": "coverage_rsrp",
        "label": "Coverage (RSRP)",
        "analysis_kind": "degradation",
        "metric_group": "coverage",
        "metric_name": "RSRP",
        "phrases": ("cells couverture rsrp", "coverage rsrp", "couverture rsrp"),
        "tokens": ("couverture", "rsrp"),
    },
    {
        "key": "quality_sinr",
        "label": "Quality (SINR)",
        "analysis_kind": "degradation",
        "metric_group": "quality",
        "metric_name": "SINR",
        "phrases": ("cells qualite sinr", "cells quality sinr", "qualite sinr", "quality sinr"),
        "tokens": ("qualite", "sinr"),
    },
    {
        "key": "coverage_rscp",
        "label": "Coverage (RSCP)",
        "analysis_kind": "degradation",
        "metric_group": "coverage",
        "metric_name": "RSCP",
        "phrases": (
            "cells couverture rscp",
            "coverage rscp",
            "couverture rscp",
            "best rscp in active set in connect state",
            "best rscp in active set",
        ),
        "tokens": ("rscp",),
    },
    {
        "key": "quality_ecno",
        "label": "Quality (EcNo)",
        "analysis_kind": "degradation",
        "metric_group": "quality",
        "metric_name": "EcNo",
        "phrases": (
            "cells qualite ecno",
            "cells qualite ec no",
            "quality ecno",
            "qualite ecno",
            "qualite ec no",
            "best ec io in active set in connect state",
            "best ec io in active set",
        ),
        "tokens": ("ec", "io"),
    },
    {
        "key": "quality_rsrq",
        "label": "Quality (RSRQ)",
        "analysis_kind": "degradation",
        "metric_group": "quality",
        "metric_name": "RSRQ",
        "phrases": ("cells qualite rsrq", "quality rsrq", "qualite rsrq"),
        "tokens": ("qualite", "rsrq"),
    },
    {
        "key": "throughput_dl",
        "label": "Débit DL",
        "analysis_kind": "degradation",
        "metric_group": "throughput",
        "metric_name": "DL Throughput",
        "phrases": ("debit dl en mobilite", "throughput dl en mobility"),
        "tokens": ("debit", "dl", "mobilite"),
    },
    {
        "key": "throughput_ul",
        "label": "Débit UL",
        "analysis_kind": "degradation",
        "metric_group": "throughput",
        "metric_name": "UL Throughput",
        "phrases": ("debit ul en mobilite", "throughput ul en mobility"),
        "tokens": ("debit", "ul", "mobilite"),
    },
)

THROUGHPUT_THRESHOLDS = {
    "L800": {"dl": 15.0, "ul": 10.0},
    "L2100": {"dl": 25.0, "ul": 15.0},
    "L2600": {"dl": 50.0, "ul": 20.0},
}


class SsvWorkbookError(ValueError):
    """Raised when the workbook cannot provide a valid SSV target image."""


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


def column_letters_to_index(column_letters: str) -> int:
    index = 0
    for char in column_letters:
        index = (index * 26) + (ord(char.upper()) - 64)
    return index - 1


def row_col_to_ref(row: int, col: int) -> str:
    col_num = col + 1
    letters = []
    while col_num:
        col_num, remainder = divmod(col_num - 1, 26)
        letters.append(chr(65 + remainder))
    return f"{''.join(reversed(letters))}{row + 1}"


def resolve_zip_path(base_path: str, target: str) -> str:
    joined = posixpath.join(posixpath.dirname(base_path), target)
    return posixpath.normpath(joined)


def parse_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in archive.namelist():
        return []

    root = ET.fromstring(archive.read(shared_strings_path))
    values: list[str] = []

    for item in root:
        fragments: list[str] = []
        for text_node in item.iter(f"{{{MAIN_NS}}}t"):
            fragments.append(text_node.text or "")
        values.append("".join(fragments).strip())

    return values


def read_sheet_cells(archive: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> dict[tuple[int, int], str]:
    root = ET.fromstring(archive.read(sheet_path))
    namespace = {"m": MAIN_NS}
    cells: dict[tuple[int, int], str] = {}

    for cell in root.findall(".//m:c", namespace):
        ref = cell.attrib.get("r", "")
        if not ref:
            continue

        match = re.match(r"([A-Z]+)(\d+)", ref)
        if not match:
            continue

        col = column_letters_to_index(match.group(1))
        row = int(match.group(2)) - 1
        cell_type = cell.attrib.get("t")
        value_node = cell.find("m:v", namespace)
        inline_node = cell.find("m:is", namespace)

        if value_node is not None:
            value = value_node.text or ""
            if cell_type == "s":
                try:
                    value = shared_strings[int(value)]
                except (ValueError, IndexError):
                    pass
        elif inline_node is not None:
            value = "".join(text_node.text or "" for text_node in inline_node.iter(f"{{{MAIN_NS}}}t"))
        else:
            continue

        cells[(row, col)] = str(value).strip()

    return cells


def read_workbook_sheets(archive: zipfile.ZipFile, shared_strings: list[str]) -> list[WorkbookSheet]:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: f"xl/{rel.attrib['Target'].lstrip('/')}"
        for rel in rel_root
        if "Id" in rel.attrib and "Target" in rel.attrib
    }
    namespace = {"m": MAIN_NS}
    sheets: list[WorkbookSheet] = []

    for sheet in workbook_root.find("m:sheets", namespace) or []:
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(f"{{{REL_NS}}}id")
        if not rel_id or rel_id not in rel_map:
            continue

        sheet_path = rel_map[rel_id].replace("xl//", "xl/")
        drawing_path = extract_sheet_drawing_path(archive, sheet_path)
        score = rank_sheet_name(name)
        cells = read_sheet_cells(archive, sheet_path, shared_strings)
        sheets.append(
            WorkbookSheet(
                name=name,
                path=sheet_path,
                drawing_path=drawing_path,
                cells=cells,
                score=score,
            )
        )

    return sheets


def extract_sheet_drawing_path(archive: zipfile.ZipFile, sheet_path: str) -> str | None:
    sheet_root = ET.fromstring(archive.read(sheet_path))
    drawing = sheet_root.find(f"{{{MAIN_NS}}}drawing")
    if drawing is None:
        return None

    drawing_rel_id = drawing.attrib.get(f"{{{REL_NS}}}id")
    if not drawing_rel_id:
        return None

    rels_path = posixpath.join(posixpath.dirname(sheet_path), "_rels", f"{posixpath.basename(sheet_path)}.rels")
    if rels_path not in archive.namelist():
        return None

    rel_root = ET.fromstring(archive.read(rels_path))
    for rel in rel_root:
        if rel.attrib.get("Id") == drawing_rel_id:
            return resolve_zip_path(sheet_path, rel.attrib.get("Target", ""))

    return None


def rank_sheet_name(sheet_name: str) -> float:
    normalized = normalize_text(sheet_name)
    score = 0.0

    if "l800" in normalized or "800" in normalized:
        score += 24.0
    if "dt" in normalized:
        score += 18.0
    if "mobilite" in normalized or "mobile" in normalized or "mobility" in normalized:
        score += 18.0
    if normalized.startswith("3 "):
        score += 8.0

    hint_similarity = max(SequenceMatcher(None, normalized, hint).ratio() for hint in TARGET_SHEET_HINTS)
    score += hint_similarity * 35.0
    return score


def extract_image_candidates(archive: zipfile.ZipFile, sheets: list[WorkbookSheet]) -> list[ImageCandidate]:
    candidates: list[ImageCandidate] = []
    for sheet in sheets:
        if not sheet.drawing_path:
            continue

        candidates.extend(build_candidates_for_sheet(archive, sheet))

    return candidates


def build_candidates_for_sheet(archive: zipfile.ZipFile, sheet: WorkbookSheet) -> list[ImageCandidate]:
    drawing_path = sheet.drawing_path
    if not drawing_path:
        return []

    drawing_rels_path = posixpath.join(posixpath.dirname(drawing_path), "_rels", f"{posixpath.basename(drawing_path)}.rels")
    if drawing_path not in archive.namelist() or drawing_rels_path not in archive.namelist():
        return []

    drawing_root = ET.fromstring(archive.read(drawing_path))
    rel_root = ET.fromstring(archive.read(drawing_rels_path))
    rel_map = {
        rel.attrib["Id"]: resolve_zip_path(drawing_path, rel.attrib["Target"])
        for rel in rel_root
        if "Id" in rel.attrib and "Target" in rel.attrib
    }

    caption_cells = []
    for (row, col), value in sheet.cells.items():
        profile_match = match_target_profile(value)
        if profile_match:
            caption_cells.append((row, col, value, profile_match))

    namespace = {"xdr": DRAW_NS, "a": A_NS}
    anchors = drawing_root.findall("xdr:twoCellAnchor", namespace)
    anchors_data = []
    for anchor in anchors:
        from_node = anchor.find("xdr:from", namespace)
        blip_node = anchor.find(".//a:blip", namespace)
        if from_node is None or blip_node is None:
            continue

        row_node = from_node.find("xdr:row", namespace)
        col_node = from_node.find("xdr:col", namespace)
        embed_rel_id = blip_node.attrib.get(f"{{{REL_NS}}}embed")
        media_path = rel_map.get(embed_rel_id or "")
        if row_node is None or col_node is None or not media_path:
            continue

        anchor_row = int(row_node.text or 0)
        anchor_col = int(col_node.text or 0)
        nearby_text, nearby_score = gather_nearby_text(sheet.cells, anchor_row, anchor_col)
        anchors_data.append(
            {
                "row": anchor_row,
                "col": anchor_col,
                "media_path": media_path,
                "nearby_text": nearby_text,
                "nearby_score": nearby_score,
            }
        )

    candidates: list[ImageCandidate] = []
    for row, col, value, profile_match in caption_cells:
        closest_anchor = None
        closest_distance = None
        for anchor in anchors_data:
            distance = abs(anchor["row"] - row) + abs(anchor["col"] - col)
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_anchor = anchor

        if closest_anchor is None:
            continue

        candidate_score = (
            (sheet.score * 4.0)
            + closest_anchor["nearby_score"]
            + max(0.0, 180.0 - ((closest_distance or 0) * 32.0))
            + (profile_match["strength"] * 20.0)
        )
        candidates.append(
            ImageCandidate(
                sheet_name=sheet.name,
                sheet_path=sheet.path,
                drawing_path=drawing_path,
                media_path=closest_anchor["media_path"],
                target_key=profile_match["key"],
                target_label=profile_match["label"],
                anchor_row=closest_anchor["row"],
                anchor_col=closest_anchor["col"],
                score=candidate_score,
                analysis_kind=str(profile_match.get("analysis_kind", "cross")),
                metric_group=str(profile_match["metric_group"]) if profile_match.get("metric_group") else None,
                metric_name=str(profile_match["metric_name"]) if profile_match.get("metric_name") else None,
                nearby_text=closest_anchor["nearby_text"],
                caption_ref=row_col_to_ref(row, col),
            )
        )

    return deduplicate_candidates(candidates)


def gather_nearby_text(cells: dict[tuple[int, int], str], anchor_row: int, anchor_col: int) -> tuple[list[str], float]:
    score = 0.0
    nearby_text: list[str] = []

    for (row, col), value in cells.items():
        normalized = normalize_text(value)
        if not normalized:
            continue

        distance = abs(row - anchor_row) + abs(col - anchor_col)
        if distance > 10:
            continue

        text_strength = target_caption_strength(value)
        if text_strength <= 0:
            continue

        weight = 1.0 / (1.0 + distance)
        score += text_strength * 12.0 * weight
        nearby_text.append(f"{row_col_to_ref(row, col)}: {value}")

    nearby_text.sort()
    return nearby_text[:6], score


def match_target_profile(value: str) -> dict[str, str | float] | None:
    normalized = normalize_text(value)
    if not normalized:
        return None

    best_match = None
    best_strength = 0.0
    for profile in TARGET_IMAGE_PROFILES:
        strength = 0.0
        for phrase in profile["phrases"]:
            if phrase in normalized:
                strength += 6.0
        for token in profile["tokens"]:
            if token in normalized:
                strength += 1.0
        if strength > best_strength:
            best_strength = strength
            best_match = {
                "key": profile["key"],
                "label": profile["label"],
                "analysis_kind": profile.get("analysis_kind", "cross"),
                "metric_group": profile.get("metric_group"),
                "metric_name": profile.get("metric_name"),
                "strength": strength,
            }

    if best_match and best_strength >= 4.0:
        return best_match
    return None


def target_caption_strength(value: str) -> float:
    match = match_target_profile(value)
    return float(match["strength"]) if match else 0.0


def caption_distance_score(anchor_row: int, anchor_col: int, caption_cells: list[tuple[int, int, str]]) -> tuple[float, str | None]:
    if not caption_cells:
        return 0.0, None

    best_distance = None
    best_ref = None
    for row, col, _value in caption_cells:
        distance = abs(anchor_row - row) + abs(anchor_col - col)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_ref = row_col_to_ref(row, col)

    if best_distance is None:
        return 0.0, None

    return max(0.0, 180.0 - (best_distance * 32.0)), best_ref


def open_workbook(file_bytes: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(file_bytes))


def deduplicate_candidates(candidates: list[ImageCandidate]) -> list[ImageCandidate]:
    deduped: dict[str, ImageCandidate] = {}
    for candidate in candidates:
        key = candidate.target_key
        current = deduped.get(key)
        if current is None or candidate.score > current.score:
            deduped[key] = candidate
    return list(deduped.values())


def select_target_images(file_bytes: bytes) -> list[tuple[ImageCandidate, bytes, str]]:
    with open_workbook(file_bytes) as archive:
        shared_strings = parse_shared_strings(archive)
        sheets = read_workbook_sheets(archive, shared_strings)
        ranked_sheets = [sheet for sheet in sheets if sheet.drawing_path]
        candidates = extract_image_candidates(archive, ranked_sheets)
        if not candidates:
            raise SsvWorkbookError(
                "No valid SSV image was found. Expected captions like Serving PCI, Serving Cell ID, Best Server, RxLev, RxQual, RSRP, or SINR."
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        if candidates[0].score < 120.0:
            raise SsvWorkbookError("The workbook images could not be matched reliably to an SSV caption.")

        selected: list[tuple[ImageCandidate, bytes, str]] = []
        for candidate in candidates:
            if candidate.media_path not in archive.namelist():
                continue
            image_bytes = archive.read(candidate.media_path)
            mime_type = mimetypes.guess_type(candidate.media_path)[0] or detect_image_mime(image_bytes)
            selected.append((candidate, image_bytes, mime_type))
            LOGGER.info(
                "Selected SSV image candidate sheet=%s label=%s media=%s score=%.2f anchor=%s nearby=%s",
                candidate.sheet_name,
                candidate.target_label,
                candidate.media_path,
                candidate.score,
                row_col_to_ref(candidate.anchor_row, candidate.anchor_col),
                candidate.nearby_text,
            )

        if not selected:
            raise SsvWorkbookError("The selected workbook image could not be extracted.")

        return selected


def select_target_image(file_bytes: bytes) -> tuple[ImageCandidate, bytes, str]:
    return select_target_images(file_bytes)[0]


def detect_image_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "application/octet-stream"


def infer_lte_band(sheet_name: str | None, filename: str | None = None) -> str | None:
    for text in (normalize_text(sheet_name or ""), normalize_text(filename or "")):
        if "l800" in text or "800" in text:
            return "L800"
        if "l2100" in text or "2100" in text:
            return "L2100"
        if "l2600" in text or "2600" in text:
            return "L2600"
    return None


def parse_numeric_mbps(value: str) -> float | None:
    cleaned = str(value or "").replace(",", ".")
    match = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def find_neighbor_metric_value(
    cells: dict[tuple[int, int], str],
    row: int,
    col: int,
) -> tuple[float | None, str | None]:
    for row_offset, col_offset in ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (-1, 1)):
        candidate_row = row + row_offset
        candidate_col = col + col_offset
        candidate_value = cells.get((candidate_row, candidate_col), "")
        parsed = parse_numeric_mbps(candidate_value)
        if parsed is not None:
            return parsed, row_col_to_ref(candidate_row, candidate_col)
    return None, None


def extract_avg_throughput_metrics(file_bytes: bytes, filename: str) -> dict[str, object] | None:
    with open_workbook(file_bytes) as archive:
        shared_strings = parse_shared_strings(archive)
        sheets = read_workbook_sheets(archive, shared_strings)

    best_result: dict[str, object] | None = None
    best_score = -1.0
    for sheet in sheets:
        dl_label_ref = None
        ul_label_ref = None
        dl_value = None
        ul_value = None
        dl_value_ref = None
        ul_value_ref = None

        for (row, col), value in sheet.cells.items():
            normalized = normalize_text(value)
            if "avg dl throughput mbps" in normalized:
                dl_label_ref = row_col_to_ref(row, col)
                dl_value, dl_value_ref = find_neighbor_metric_value(sheet.cells, row, col)
            elif "avg ul throughput mbps" in normalized:
                ul_label_ref = row_col_to_ref(row, col)
                ul_value, ul_value_ref = find_neighbor_metric_value(sheet.cells, row, col)

        if dl_value is None or ul_value is None:
            continue

        band = infer_lte_band(sheet.name, filename)
        thresholds = THROUGHPUT_THRESHOLDS.get(band or "")
        score = sheet.score + (50.0 if band else 0.0)
        if score <= best_score:
            continue

        best_result = {
            "sheet_name": sheet.name,
            "band": band,
            "dl_value_mbps": dl_value,
            "ul_value_mbps": ul_value,
            "dl_threshold_mbps": thresholds["dl"] if thresholds else None,
            "ul_threshold_mbps": thresholds["ul"] if thresholds else None,
            "dl_label_ref": dl_label_ref,
            "ul_label_ref": ul_label_ref,
            "dl_value_ref": dl_value_ref,
            "ul_value_ref": ul_value_ref,
        }
        best_score = score

    return best_result
