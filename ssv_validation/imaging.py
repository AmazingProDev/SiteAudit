from __future__ import annotations

import logging
import struct
import subprocess
from pathlib import Path

from .models import Bitmap

LOGGER = logging.getLogger(__name__)


class SsvImageError(ValueError):
    """Raised when the SSV image cannot be prepared for analysis."""


def convert_image_to_bmp(source_path: Path, output_path: Path) -> None:
    try:
        completed = subprocess.run(
            ["sips", "-s", "format", "bmp", str(source_path), "--out", str(output_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        LOGGER.debug("sips output: %s", completed.stdout.strip())
    except FileNotFoundError as exc:
        raise SsvImageError("The macOS 'sips' tool is required to analyze SSV images on this machine.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        raise SsvImageError(f"The embedded image could not be normalized for analysis: {stderr or 'sips failed'}") from exc


def decode_bmp(bitmap_path: Path) -> Bitmap:
    data = bitmap_path.read_bytes()
    if data[:2] != b"BM":
        raise SsvImageError("The normalized SSV image is not a BMP file.")

    pixel_offset = struct.unpack("<I", data[10:14])[0]
    dib_size = struct.unpack("<I", data[14:18])[0]
    width, height = struct.unpack("<ii", data[18:26])
    top_down = height < 0
    height = abs(height)
    bits_per_pixel = struct.unpack("<H", data[28:30])[0]
    compression = struct.unpack("<I", data[30:34])[0]

    if bits_per_pixel not in (24, 32):
        raise SsvImageError(f"Unsupported BMP pixel depth: {bits_per_pixel}")

    pixels: list[list[tuple[int, int, int]]] = []

    if bits_per_pixel == 32:
        row_stride = width * 4
        for y in range(height):
            row_index = y if top_down else (height - 1 - y)
            base = pixel_offset + (row_index * row_stride)
            row: list[tuple[int, int, int]] = []
            for x in range(width):
                blue, green, red, _alpha = data[base + x * 4 : base + x * 4 + 4]
                row.append((red, green, blue))
            pixels.append(row)
    else:
        row_stride = ((width * 3) + 3) & ~3
        for y in range(height):
            row_index = y if top_down else (height - 1 - y)
            base = pixel_offset + (row_index * row_stride)
            row = []
            for x in range(width):
                blue, green, red = data[base + x * 3 : base + x * 3 + 3]
                row.append((red, green, blue))
            pixels.append(row)

    if compression not in (0, 3):
        raise SsvImageError(f"Unsupported BMP compression mode: {compression}")

    if dib_size < 40:
        raise SsvImageError("The normalized BMP header is incomplete.")

    return Bitmap(width=width, height=height, pixels=pixels)

