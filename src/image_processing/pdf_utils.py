"""Shared PDF/image utilities for DPI computation and image selection.

Used by ``scripts/convert_sources.py`` and ``scripts/compute_dpi.py``.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pymupdf
from PIL import Image

# ---------------------------------------------------------------------------
# DPI helpers
# ---------------------------------------------------------------------------

# Sane range for scanned historical documents
DPI_MIN = 72
DPI_MAX = 1200


def round_dpi(value: float, step: int = 25) -> int:
    """Round a DPI value to the nearest *step* (default 25).

    Returns at least ``DPI_MIN``.
    """
    rounded = int(round(value / step) * step)
    return max(rounded, DPI_MIN)


def clamp_dpi(value: float, label: str = "") -> float:
    """Warn and clamp a DPI value to [DPI_MIN, DPI_MAX].

    Returns the original value if within range, otherwise the nearest
    bound with a printed warning.
    """
    if value < DPI_MIN:
        ctx = f" ({label})" if label else ""
        print(f"  Warning: DPI {value:.0f}{ctx} below minimum, clamping to {DPI_MIN}")
        return float(DPI_MIN)
    if value > DPI_MAX:
        ctx = f" ({label})" if label else ""
        print(f"  Warning: DPI {value:.0f}{ctx} above maximum, clamping to {DPI_MAX}")
        return float(DPI_MAX)
    return value


# ---------------------------------------------------------------------------
# PDF image selection
# ---------------------------------------------------------------------------

def select_best_image(
    page: pymupdf.Page,
    doc: pymupdf.Document,
) -> tuple[int, int, int, list[int]]:
    """Pick the best scan image from a PDF page.

    Strategy for scanned documents:
      1. Collect all xrefs used as soft-masks — those are alpha masks,
         not standalone content images.
      2. Among the remaining (non-mask) images, choose the one with the
         largest pixel area (width × height).

    Returns
    -------
    best_xref : int
        The xref of the selected image (0 if none found).
    best_w : int
        Width in pixels of the selected image (0 if none found).
    best_h : int
        Height in pixels of the selected image (0 if none found).
    mask_xrefs : list[int]
        Xrefs of images that were identified as masks.
    """
    images = page.get_images(full=True)
    if not images:
        return 0, 0, 0, []

    # images entry: (xref, smask, width, height, bpc, colorspace, ...)
    # Collect xrefs used as soft-masks by other images
    mask_xrefs: set[int] = set()
    for entry in images:
        smask_xref = entry[1]
        if smask_xref != 0:
            mask_xrefs.add(smask_xref)

    # Filter to non-mask candidate images
    candidates = [
        entry for entry in images if entry[0] not in mask_xrefs
    ]
    if not candidates:
        # Fallback: all images are masks — just use all of them
        candidates = list(images)

    # Pick the largest by pixel area
    best = max(candidates, key=lambda e: e[2] * e[3])
    return best[0], best[2], best[3], list(mask_xrefs)


# ---------------------------------------------------------------------------
# TIFF DPI with resolution-unit awareness
# ---------------------------------------------------------------------------

def get_tiff_dpi(img: Image.Image) -> tuple[float | None, float | None]:
    """Extract DPI from a TIFF image, handling resolution units correctly.

    Pillow's ``img.info["dpi"]`` returns raw values without converting
    units.  TIFF tag 296 (ResolutionUnit) can be:
      - 1 = No unit (undefined)
      - 2 = Dots per inch (default)
      - 3 = Dots per centimeter

    Returns (dpi_w, dpi_h) in dots-per-inch, or (None, None) if missing.
    """
    dpi = img.info.get("dpi", (None, None))
    dpi_w, dpi_h = dpi
    if dpi_w is None and dpi_h is None:
        return None, None

    # Check resolution unit via TIFF tag
    res_unit = 2  # default = inches
    try:
        # tag_v2 is the modern TIFF tag dict in Pillow
        res_unit = img.tag_v2.get(296, 2)
    except AttributeError:
        try:
            res_unit = img.tag.get(296, (2,))[0]
        except (AttributeError, IndexError, TypeError):
            pass

    if res_unit == 3:
        # Convert dots-per-cm → dots-per-inch
        if dpi_w is not None:
            dpi_w = float(dpi_w) * 2.54
        if dpi_h is not None:
            dpi_h = float(dpi_h) * 2.54
    elif res_unit == 1:
        # No unit — values are unreliable
        return None, None

    return (float(dpi_w) if dpi_w else None, float(dpi_h) if dpi_h else None)


# ---------------------------------------------------------------------------
# Aggregate DPI from a set of values
# ---------------------------------------------------------------------------

def median_dpi(values: list[float], step: int = 25) -> float:
    """Return the median of *values*, rounded to nearest *step*."""
    if not values:
        return 150.0  # fallback
    med = statistics.median(values)
    return float(round_dpi(med, step))
