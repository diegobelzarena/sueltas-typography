"""Shared PDF/image utilities for DPI computation and image selection.

Used by ``scripts/convert_sources.py`` and ``scripts/compute_dpi.py``.

Enhanced API (per-page scan-image selector and DPI estimator)
-------------------------------------------------------------
* :func:`select_best_scan_image` — pick the image most likely to be the
  full scanned page, using **bbox coverage** as the primary heuristic.
* :func:`estimate_effective_dpi` — compute effective DPI from image pixel
  dimensions and the placement bounding box.
* :func:`render_page_to_target_dpi` — rasterise a PDF page via PyMuPDF's
  matrix renderer.
* :func:`extract_scanned_page_image` — high-level convenience that combines
  selection, DPI estimation, extraction, and rescaling.

Why bbox coverage beats raw pixel area
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A large embedded thumbnail or an off-page decorative image may have many
pixels but cover only a small fraction of the visible page area.  Coverage
directly measures "does this image fill the page?" and is therefore a better
proxy for *is this the scanned page?* than raw pixel count.  Example: a
3 000 × 4 000 px image placed at 1 % of the page is probably a caption
illustration; a 600 × 800 px image placed at 95 % of the page almost
certainly *is* the scan.
"""

from __future__ import annotations

from io import BytesIO
import statistics
from pathlib import Path
from typing import TypedDict

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
# Enhanced scan-image selection with bbox coverage
# ---------------------------------------------------------------------------

#: Minimum fraction of page area that a candidate image bbox must cover.
MIN_COVERAGE: float = 0.50

#: Minimum pixel area to accept a candidate (rejects tiny thumbnails/icons).
MIN_PIXEL_AREA: int = 10_000  # ≈ 100 × 100 px


class ScanImageResult(TypedDict):
    """Structured result from :func:`select_best_scan_image`."""

    xref: int
    width: int           # pixels
    height: int          # pixels
    pixel_area: int
    coverage: float      # fraction of page area covered by image bbox
    bbox: tuple[float, float, float, float] | None  # (x0, y0, x1, y1) in pts


def _image_coverage(
    page: pymupdf.Page,
    name: str,
) -> tuple[float, tuple[float, float, float, float] | None]:
    """Return (coverage_fraction, bbox_tuple) for an image placed on *page*.

    Coverage = area(clip(image_bbox, page_rect)) / area(page_rect).

    Returns ``(0.0, None)`` when the image has no placement rect (e.g. an
    inline image or an unreferenced form XObject).
    """
    try:
        rects = page.get_image_rects(name)
    except Exception:
        return 0.0, None

    if not rects:
        return 0.0, None

    # Union of all placements (handles images tiled/repeated on one page)
    union: pymupdf.Rect = rects[0]
    for r in rects[1:]:
        union = union | r

    # Clip to page rect and measure fraction covered
    clipped = union & page.rect
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return 0.0, None

    coverage = (clipped.width * clipped.height) / page_area
    bbox = (union.x0, union.y0, union.x1, union.y1)
    return coverage, bbox


def _rank_scan_candidates(
    candidates: list[dict],
    min_coverage: float = MIN_COVERAGE,
    min_pixel_area: int = MIN_PIXEL_AREA,
) -> list[dict]:
    """Filter and rank image candidate dicts; prefer non-JBIG2, then coverage, then pixel_area.

    Each dict in *candidates* must have keys: ``xref``, ``width``,
    ``height``, ``pixel_area``, ``coverage``, ``is_jbig2``.

    Rejection criteria (false-positive filters)
    --------------------------------------------
    * **coverage < min_coverage** — image bbox covers less than *min_coverage*
      of the page; likely a thumbnail, logo, or off-page ornament.
    * **pixel_area < min_pixel_area** — too few pixels to be a scan (e.g. a
      watermark glyph or navigation icon).

    Fallback strategy
    -----------------
    If any non-JBIG2 candidates exist, JBIG2 candidates are ignored.
    Within that preferred pool, if no candidate passes *both* thresholds the
    filter relaxes:

    1. Coverage-only (pixel_area filter dropped).
    2. Unconditional — all candidates are returned sorted by
       ``(coverage DESC, pixel_area DESC)``.

    This ensures the function always returns a result when images are present.
    """

    def sort_key(c: dict) -> tuple[float, int]:
        return (c["coverage"], c["pixel_area"])

    non_jbig2_candidates = [c for c in candidates if not c["is_jbig2"]]
    candidate_pool = non_jbig2_candidates or candidates

    qualified = [
        c for c in candidate_pool
        if c["coverage"] >= min_coverage and c["pixel_area"] >= min_pixel_area
    ]
    if not qualified:
        # Relax: drop pixel_area filter
        qualified = [c for c in candidate_pool if c["coverage"] >= min_coverage]
    if not qualified:
        # Full fallback
        qualified = list(candidate_pool)

    return sorted(qualified, key=sort_key, reverse=True)


def select_best_scan_image(
    page: pymupdf.Page,
    doc: pymupdf.Document,
    min_coverage: float = MIN_COVERAGE,
    min_pixel_area: int = MIN_PIXEL_AREA,
) -> ScanImageResult | None:
    """Select the image most likely to be the full scanned page.

    Selection heuristic (priority order)
    -------------------------------------
    1. **Exclude soft-masks** — images referenced as ``smask`` by another
       image are alpha channels, not content images.
    2. **Compute bbox coverage** — for each remaining image, measure what
       fraction of the page rect its placement bbox covers.
    3. **Reject false positives** — discard images whose coverage is below
       *min_coverage* (default 50 %) or whose pixel area is below
       *min_pixel_area* (default 10 000 px²).
    4. **Rank by (coverage DESC, pixel_area DESC)** — prefer the image that
       fills the most page area; break ties with raw pixel count.
    5. **Graceful fallback** — if no image passes the thresholds, relax
       progressively so the function always returns a result when images exist.

    Failure modes (threat model)
    ----------------------------
    * **Multi-layer PDFs** — background + text layer: only the background is
      returned; the text layer is masked out by the smask filter.
    * **Thumbnail strip** — PDFs may embed a small thumbnail alongside the
      full scan.  The coverage filter eliminates it (thumbnail bbox is tiny).
    * **Off-page images** — placed outside the page rect; near-zero clipped
      coverage, filtered out.
    * **Rotated pages** — ``page.rect`` reflects displayed dimensions, so
      coverage is always in display space regardless of rotation.
    * **Missing placement** — if ``get_image_rects`` returns nothing (inline
      images or unreferenced XObjects), coverage = 0.0 and the image is only
      selected via the unconditional fallback path.
    * **All-mask pages** — when every listed image is a soft-mask, the largest
      mask is returned as a last resort.

    Returns ``None`` if the page contains no images at all.
    """
    images = page.get_images(full=True)
    if not images:
        return None

    # images entry: (xref, smask, width, height, bpc, colorspace, alt_cs, name, filter, ...)
    mask_xrefs: set[int] = {entry[0] for entry in images if entry[1] != 0}

    candidates: list[dict] = []
    for entry in images:
        xref = entry[0]
        w, h = entry[2], entry[3]
        name: str = entry[7] if len(entry) > 7 else str(xref)
        filter_name = entry[8] if len(entry) > 8 else ""
        if xref in mask_xrefs:
            continue
        coverage, bbox = _image_coverage(page, entry)
        candidates.append({
            "xref": xref,
            "width": w,
            "height": h,
            "pixel_area": w * h,
            "coverage": coverage,
            "bbox": bbox,
            "is_jbig2": str(filter_name).lstrip("/").lower() == "jbig2decode",
            "name": name,
        })

    if not candidates:
        # All images were masks — fall back to the largest mask
        best_entry = max(images, key=lambda e: e[2] * e[3])
        xref = best_entry[0]
        w, h = best_entry[2], best_entry[3]
        name = best_entry[7] if len(best_entry) > 7 else str(xref)
        coverage, bbox = _image_coverage(page, best_entry)
        return ScanImageResult(
            xref=xref, width=w, height=h, pixel_area=w * h,
            coverage=coverage, bbox=bbox,
        )

    ranked = _rank_scan_candidates(candidates, min_coverage, min_pixel_area)
    best = ranked[0]
    return ScanImageResult(
        xref=best["xref"],
        width=best["width"],
        height=best["height"],
        pixel_area=best["pixel_area"],
        coverage=best["coverage"],
        bbox=best["bbox"],
    )


def estimate_effective_dpi(
    result: ScanImageResult,
    page: pymupdf.Page,
) -> tuple[float | None, float | None]:
    """Compute effective DPI from image pixel dimensions and placement bbox.

    Formula::

        dpi_x = image_width_px  / bbox_width_pts  * 72
        dpi_y = image_height_px / bbox_height_pts * 72

    where *bbox* is the image's placement rectangle in PDF points.  When no
    placement bbox is available the function falls back to the full page
    dimensions.

    Each axis is computed independently; an axis whose dimension is zero
    returns ``None`` for that component.  Both values are clamped to
    ``[DPI_MIN, DPI_MAX]``.

    Returns ``(None, None)`` when page dimensions are also unavailable.
    """
    w_px, h_px = result["width"], result["height"]
    bbox = result.get("bbox")

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        w_pts = abs(x1 - x0)
        h_pts = abs(y1 - y0)
    else:
        w_pts = page.rect.width
        h_pts = page.rect.height

    dpi_x = clamp_dpi(w_px / w_pts * 72) if w_pts > 0 else None
    dpi_y = clamp_dpi(h_px / h_pts * 72) if h_pts > 0 else None
    return dpi_x, dpi_y


def render_page_to_target_dpi(
    page: pymupdf.Page,
    target_dpi: int = 150,
) -> pymupdf.Pixmap:
    """Rasterise *page* to a :class:`pymupdf.Pixmap` at *target_dpi*.

    Uses PyMuPDF's matrix-based renderer.  Zoom factor::

        zoom = target_dpi / 72

    since one PDF point = 1/72 inch.
    """
    zoom = target_dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    return page.get_pixmap(matrix=mat, alpha=False)


def _pixmap_to_pil(pix: pymupdf.Pixmap) -> Image.Image:
    """Convert a :class:`pymupdf.Pixmap` to a PIL :class:`~PIL.Image.Image`.

    Handles CMYK (4-channel) pixmaps by converting to RGB first to avoid
    channel misinterpretation.
    """
    if pix.colorspace and pix.colorspace.n == 4 and not pix.alpha:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    channel_count = pix.colorspace.n if pix.colorspace else pix.n - (1 if pix.alpha else 0)
    if pix.alpha:
        mode = {1: "LA", 3: "RGBA"}.get(channel_count)
    else:
        mode = {1: "L", 3: "RGB"}.get(channel_count)

    if mode is not None:
        try:
            return Image.frombytes(mode, (pix.width, pix.height), bytes(pix.samples))
        except (ValueError, OSError):
            pass

    with Image.open(BytesIO(pix.tobytes("png"))) as img:
        return img.copy()


def extract_scanned_page_image(
    page: pymupdf.Page,
    doc: pymupdf.Document,
    target_dpi: int = 150,
    min_coverage: float = MIN_COVERAGE,
    min_pixel_area: int = MIN_PIXEL_AREA,
) -> Image.Image | None:
    """Extract the scanned page as a PIL Image scaled to *target_dpi*.

    Pipeline
    --------
    1. :func:`select_best_scan_image` — identify the scan image.
    2. :func:`estimate_effective_dpi` — determine source resolution.
    3. Extract the raw embedded image via :class:`pymupdf.Pixmap`.
    4. Scale to *target_dpi* with LANCZOS resampling (skipped when the
       source and target DPI differ by ≤ 2 %).

    Falls back to :func:`render_page_to_target_dpi` (full page rasterisation)
    when no suitable embedded image is found (e.g. vector-only PDF pages).

    Returns ``None`` only if the page is completely blank.
    """
    result = select_best_scan_image(page, doc, min_coverage, min_pixel_area)

    if result is None:
        # No images — render the page as vector graphics
        pix = render_page_to_target_dpi(page, target_dpi)
        return _pixmap_to_pil(pix) if pix.width > 0 else None

    pix = pymupdf.Pixmap(doc, result["xref"])
    img = _pixmap_to_pil(pix)

    src_dpi_x, src_dpi_y = estimate_effective_dpi(result, page)
    src_dpi = src_dpi_x or src_dpi_y or 72.0

    if abs(src_dpi - target_dpi) / max(src_dpi, 1) > 0.02:
        scale = target_dpi / src_dpi
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    return img


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
