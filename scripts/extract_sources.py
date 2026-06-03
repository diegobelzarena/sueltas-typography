#!/usr/bin/env python
"""Pipeline for source inspection and page-image extraction from PDFs and TIFF folders.

Pipeline
--------
1. Discover PDF files and TIFF folders.
2. Inspect each source once and build per-page observations.
3. Extract one image per page.
4. Normalize the output to a target DPI.
5. Save PNGs and a lightweight report.

Notes
-----
* The TIFF path is implemented as a reference implementation.
* The PDF path is intentionally left as two hooks:
  - inspect_pdf_source
  - extract_pdf_page_image

Those hooks are where collection-specific PDF inspection and exception
handling should live.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageFilter
import pymupdf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from image_processing.pdf_utils import _pixmap_to_pil, clamp_dpi, get_tiff_dpi, render_page_to_target_dpi, round_dpi
from shared.tools.report import StepReport


SourceKind = Literal["pdf", "tiff_folder"]


# First-pass grouping for "looks like the full page width".
PDF_RECT_WIDTH_REL_TOL = 0.03
PDF_RECT_WIDTH_ABS_TOL_PTS = 12.0
PDF_MIN_RECT_HEIGHT_RATIO = 0.10
PDF_MAX_ASPECT_RATIO = 8.0
PDF_BBOX_POS_TOL_PTS = 2.0
PDF_BBOX_SIZE_REL_TOL = 0.02
PDF_VERTICAL_SPLIT_MIN_COVERAGE = 0.85
PDF_BINARY_BLUR_RADIUS = 0.6
EXPECTED_OUTPUT_HEIGHT_PX = 1230
OUTPUT_HEIGHT_REL_TOL = 0.16
DEFAULT_SOURCE_DPI = 72.0
DEFAULT_SOURCE_DPI_ABS_TOL = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the extraction pipeline."""

    target_dpi: int = 150
    skip_existing: bool = False


@dataclass(frozen=True)
class SourceRef:
    """A single input source to inspect and extract."""

    name: str
    path: Path
    kind: SourceKind


@dataclass
class PageInspection:
    """Observation produced during inspection, before extraction.

    The `payload` field is intentionally generic. For TIFFs it stores the page
    path. For PDFs it can later store whatever the extraction stage needs:
    page index, xref, bbox, chosen strategy, etc.
    """

    page_index: int
    output_name: str
    extraction_mode: str
    source_dpi: float | None = None
    source_width_px: int | None = None
    source_height_px: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SourceResult:
    """Per-source outcome, ready to serialize into the step report."""

    source: SourceRef
    status: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reason: str | None = None

    def to_report_dict(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "source_type": self.source.kind,
            "status": self.status,
            "n_pages": sum(1 for page in self.pages if page.get("status") == "ok"),
        }
        if self.pages:
            info["pages"] = self.pages
        if self.reason is not None:
            info["reason"] = self.reason
        if self.warnings:
            info["warnings"] = self.warnings
        return info


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _is_tiff_folder(path: Path) -> bool:
    """Return True when *path* contains TIFF page images."""
    return path.is_dir() and any(
        item.suffix.lower() in (".tif", ".tiff")
        for item in path.iterdir()
    )


def discover_sources(root: Path) -> list[SourceRef]:
    """Find all supported sources under *root*."""
    sources: list[SourceRef] = []
    for item in sorted(root.iterdir()):
        if item.is_file() and item.suffix.lower() == ".pdf":
            sources.append(SourceRef(name=item.stem, path=item, kind="pdf"))
        elif _is_tiff_folder(item):
            sources.append(SourceRef(name=item.name, path=item, kind="tiff_folder"))
    return sources


def source_output_dir(source: SourceRef, output_root: Path) -> Path:
    """Return the folder where extracted pages for *source* will be written."""
    return output_root / source.name


def has_existing_output(source: SourceRef, output_root: Path) -> bool:
    """Check whether the source already has saved PNG pages."""
    out_dir = source_output_dir(source, output_root)
    return out_dir.is_dir() and any(out_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def summarize_source_dpi(
    dpi_w: float | None,
    dpi_h: float | None,
    label: str,
) -> tuple[float | None, list[str]]:
    """Collapse axis DPI metadata into one source DPI and warning list."""
    values: list[float] = []
    warnings: list[str] = []

    if dpi_w is not None:
        values.append(clamp_dpi(dpi_w, f"{label}:x"))
    if dpi_h is not None:
        values.append(clamp_dpi(dpi_h, f"{label}:y"))

    if not values:
        return None, ["missing_dpi_metadata"]

    if len(values) == 2:
        disagreement = abs(values[0] - values[1]) / max(values[0], values[1], 1.0)
        if disagreement > 0.05:
            warnings.append("dpi_axes_disagree")

    return sum(values) / len(values), warnings


def _is_binary_like_image(image: Image.Image) -> bool:
    """Return True when the image contains only two tone values."""
    if image.mode == "1":
        return True

    analysis_image: Image.Image | None
    if image.mode == "L":
        analysis_image = image
    elif image.mode in ("LA", "P", "RGB", "RGBA"):
        analysis_image = image.convert("L")
    else:
        return False

    try:
        colors = analysis_image.getcolors(maxcolors=3)
        return colors is not None and len(colors) <= 2
    finally:
        if analysis_image is not image:
            analysis_image.close()


def _prepare_image_for_output(image: Image.Image) -> Image.Image:
    """Convert binary-like images to softened gray before optional resizing."""
    if not _is_binary_like_image(image):
        return image.copy()

    if image.mode == "L":
        gray_image = image.copy()
    else:
        gray_image = image.convert("L")

    try:
        return gray_image.filter(ImageFilter.GaussianBlur(radius=PDF_BINARY_BLUR_RADIUS))
    finally:
        gray_image.close()


def _is_default_source_dpi(source_dpi: float | None) -> bool:
    """Return True when the detected DPI looks like the PDF/TIFF fallback 72 DPI."""
    return source_dpi is not None and abs(source_dpi - DEFAULT_SOURCE_DPI) <= DEFAULT_SOURCE_DPI_ABS_TOL


def _maybe_correct_source_dpi(
    inspection: PageInspection | None,
    image_height_px: int,
    source_dpi: float | None,
    target_dpi: int,
) -> float | None:
    """Correct suspicious source DPI estimates using the legacy 1230px heuristic."""
    if source_dpi is None or source_dpi <= 0 or image_height_px <= 0:
        return source_dpi

    estimated_output_height = image_height_px * target_dpi / source_dpi
    output_height_error = abs(estimated_output_height - EXPECTED_OUTPUT_HEIGHT_PX) / EXPECTED_OUTPUT_HEIGHT_PX

    correction_reasons: list[str] = []
    if _is_default_source_dpi(source_dpi):
        correction_reasons.append("default_source_dpi_72")
    if output_height_error > OUTPUT_HEIGHT_REL_TOL:
        correction_reasons.append("wildly_wrong_output_height")

    if not correction_reasons:
        return source_dpi

    corrected_dpi = float(round_dpi((image_height_px * target_dpi) / EXPECTED_OUTPUT_HEIGHT_PX, step=50))
    if abs(corrected_dpi - source_dpi) <= DEFAULT_SOURCE_DPI_ABS_TOL:
        return source_dpi

    if inspection is not None:
        if "source_dpi_corrected" not in inspection.warnings:
            inspection.warnings.append("source_dpi_corrected")
        inspection.payload["dpi_correction_applied"] = True
        inspection.payload["dpi_correction_reason"] = "+".join(correction_reasons)
        inspection.payload["reported_source_dpi"] = round(source_dpi, 2)
        inspection.payload["corrected_source_dpi"] = corrected_dpi
        inspection.payload["estimated_output_height_px"] = round(estimated_output_height, 1)
        inspection.payload["expected_output_height_px"] = EXPECTED_OUTPUT_HEIGHT_PX
        inspection.source_dpi = corrected_dpi

    return corrected_dpi


def normalize_to_target_dpi(
    image: Image.Image,
    source_dpi: float | None,
    target_dpi: int,
    inspection: PageInspection | None = None,
) -> tuple[Image.Image, float, bool]:
    """Resize *image* when a source DPI is available and differs from target."""
    prepared_image = _prepare_image_for_output(image)
    effective_source_dpi = _maybe_correct_source_dpi(
        inspection,
        prepared_image.height,
        source_dpi,
        target_dpi,
    )

    if effective_source_dpi is None or effective_source_dpi <= 0:
        return prepared_image, 1.0, False

    scale_factor = target_dpi / effective_source_dpi
    if abs(scale_factor - 1.0) <= 0.01:
        return prepared_image, 1.0, False

    new_w = max(1, int(round(prepared_image.width * scale_factor)))
    new_h = max(1, int(round(prepared_image.height * scale_factor)))
    resized = prepared_image.resize((new_w, new_h), resample=Image.LANCZOS)
    prepared_image.close()
    return resized, scale_factor, True


def save_page_image(image: Image.Image, output_path: Path, target_dpi: int) -> None:
    """Save the extracted page image as PNG with embedded output DPI."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path), dpi=(target_dpi, target_dpi))


def _get_pdf_image_rects(
    page: pymupdf.Page,
    xref: int,
    name: str | None,
) -> list[pymupdf.Rect]:
    """Return placement rects for an image, trying xref first and name second."""
    try:
        rects = page.get_image_rects(xref)
        if rects:
            return rects
    except Exception:
        pass

    if name is not None:
        try:
            return page.get_image_rects(name)
        except Exception:
            pass

    return []


def _filter_small_rects(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Drop placement rects that are too short relative to their raw union height."""
    if not rects:
        return []

    raw_union = rects[0]
    for rect in rects[1:]:
        raw_union = raw_union | rect

    if raw_union.height <= 0:
        return rects

    min_height = max(raw_union.height * PDF_MIN_RECT_HEIGHT_RATIO, raw_union.width / PDF_MAX_ASPECT_RATIO)
    return [rect for rect in rects if rect.height >= min_height]


def _union_rects(rects: list[pymupdf.Rect]) -> tuple[float, float, float, float] | None:
    """Return the union bbox of all placement rects on a page."""
    if not rects:
        return None

    union = rects[0]
    for rect in rects[1:]:
        union = union | rect

    return (union.x0, union.y0, union.x1, union.y1)


def _union_bboxes(bboxes: list[tuple[float, float, float, float] | None]) -> tuple[float, float, float, float] | None:
    """Return the union bbox across several candidate bbox tuples."""
    valid_bboxes = [bbox for bbox in bboxes if bbox is not None]
    if not valid_bboxes:
        return None

    x0 = min(bbox[0] for bbox in valid_bboxes)
    y0 = min(bbox[1] for bbox in valid_bboxes)
    x1 = max(bbox[2] for bbox in valid_bboxes)
    y1 = max(bbox[3] for bbox in valid_bboxes)
    return (x0, y0, x1, y1)


def _bbox_size_at_dpi(
    bbox: tuple[float, float, float, float],
    dpi: int,
) -> tuple[int, int]:
    """Return bbox size in pixels at the requested DPI."""
    x0, y0, x1, y1 = bbox
    width_px = max(1, int(round(abs(x1 - x0) * dpi / 72.0)))
    height_px = max(1, int(round(abs(y1 - y0) * dpi / 72.0)))
    return width_px, height_px


def _bbox_offset_in_union(
    bbox: tuple[float, float, float, float],
    union_bbox: tuple[float, float, float, float],
    dpi: int,
) -> tuple[int, int]:
    """Return top-left pixel offset of *bbox* inside *union_bbox* at *dpi*."""
    x0, y0, _, _ = bbox
    ux0, uy0, _, _ = union_bbox
    offset_x = max(0, int(round((x0 - ux0) * dpi / 72.0)))
    offset_y = max(0, int(round((y0 - uy0) * dpi / 72.0)))
    return offset_x, offset_y


def _summarize_candidate_group_dpi(candidates: list[dict[str, Any]]) -> float | None:
    """Return a representative source DPI for a candidate group.

    Prefer placed image layers over PDF ImageMask stencil overlays. Stencil
    layers often have a different effective DPI and should not drive output
    scaling when an RGB/gray scan layer is present.
    """
    preferred_values = [
        candidate["source_dpi"]
        for candidate in candidates
        if candidate["source_dpi"] is not None and not candidate.get("is_image_mask", False)
    ]
    if preferred_values:
        return sum(preferred_values) / len(preferred_values)

    fallback_values = [candidate["source_dpi"] for candidate in candidates if candidate["source_dpi"] is not None]
    if not fallback_values:
        return None
    return sum(fallback_values) / len(fallback_values)


def _ordered_candidate_xrefs(candidates: list[dict[str, Any]]) -> list[int]:
    """Return candidate xrefs in PDF layer order."""
    return [candidate["xref"] for candidate in sorted(candidates, key=lambda item: item["layer_idx"])]


def _composite_candidates_share_bboxes(candidates: list[dict[str, Any]]) -> bool:
    """Return True when all candidate layers have placement bboxes for compositing."""
    return all(candidate["bbox"] is not None for candidate in candidates)


def _bboxes_are_approx_equal(
    bbox_a: tuple[float, float, float, float] | None,
    bbox_b: tuple[float, float, float, float] | None,
) -> bool:
    """Return True when two candidate bboxes are effectively the same."""
    if bbox_a is None or bbox_b is None:
        return False

    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    a_width = abs(ax1 - ax0)
    a_height = abs(ay1 - ay0)
    b_width = abs(bx1 - bx0)
    b_height = abs(by1 - by0)

    width_ref = max(a_width, b_width, 1.0)
    height_ref = max(a_height, b_height, 1.0)

    return (
        abs(ax0 - bx0) <= PDF_BBOX_POS_TOL_PTS
        and abs(ay0 - by0) <= PDF_BBOX_POS_TOL_PTS
        and abs(a_width - b_width) / width_ref <= PDF_BBOX_SIZE_REL_TOL
        and abs(a_height - b_height) / height_ref <= PDF_BBOX_SIZE_REL_TOL
    )


def _is_layered_full_page_group(candidates: list[dict[str, Any]]) -> bool:
    """Return True for full-page overlay layers sharing the same bbox.

    This covers the case where one full-page image acts as the background and
    one or more full-page overlay layers use soft masks on top of it.
    """
    if len(candidates) < 2:
        return False
    if not _composite_candidates_share_bboxes(candidates):
        return False
    if not any(candidate["has_mask"] for candidate in candidates):
        return False

    reference_bbox = candidates[0]["bbox"]
    return all(
        _bboxes_are_approx_equal(reference_bbox, candidate["bbox"])
        for candidate in candidates[1:]
    )


def _is_vertical_split_full_page_group(
    candidates: list[dict[str, Any]],
    page_height_pts: float,
) -> bool:
    """Return True when full-width candidates are stacked vertically.

    This is the special case where one logical page image is split into a top
    and bottom piece. Those pages should fall back to rendering the PDF page
    directly rather than using the candidate-composite path.
    """
    if len(candidates) < 2:
        return False
    if not _composite_candidates_share_bboxes(candidates):
        return False
    if _is_layered_full_page_group(candidates):
        return False

    sorted_bboxes = sorted(
        (candidate["bbox"] for candidate in candidates if candidate["bbox"] is not None),
        key=lambda bbox: bbox[1],
    )
    if len(sorted_bboxes) != len(candidates):
        return False

    union_bbox = _union_bboxes(sorted_bboxes)
    if union_bbox is None:
        return False

    ux0, uy0, ux1, uy1 = union_bbox
    union_width = abs(ux1 - ux0)
    union_height = abs(uy1 - uy0)
    if union_width <= 0 or union_height <= 0:
        return False

    total_segment_height = 0.0
    for bbox in sorted_bboxes:
        x0, y0, x1, y1 = bbox
        segment_width = abs(x1 - x0)
        segment_height = abs(y1 - y0)

        if abs(x0 - ux0) > PDF_BBOX_POS_TOL_PTS or abs(x1 - ux1) > PDF_BBOX_POS_TOL_PTS:
            return False
        if abs(segment_width - union_width) / max(union_width, 1.0) > PDF_BBOX_SIZE_REL_TOL:
            return False
        if segment_height >= union_height * (1.0 - PDF_BBOX_SIZE_REL_TOL):
            return False

        total_segment_height += segment_height

    for previous_bbox, current_bbox in zip(sorted_bboxes, sorted_bboxes[1:]):
        overlap = max(0.0, min(previous_bbox[3], current_bbox[3]) - max(previous_bbox[1], current_bbox[1]))
        if overlap > PDF_BBOX_POS_TOL_PTS:
            return False

    if total_segment_height / union_height < PDF_VERTICAL_SPLIT_MIN_COVERAGE:
        return False
    if page_height_pts > 0 and union_height / page_height_pts < PDF_VERTICAL_SPLIT_MIN_COVERAGE:
        return False

    return True


def _has_full_page_image_mask_with_other_candidates(
    candidates: list[dict[str, Any]],
    page_rect: pymupdf.Rect,
) -> bool:
    """Return True when a near-full-page stencil coexists with multiple image pieces.

    These pages tend to be built from many smaller placed images plus one
    full-page ImageMask stencil. Reconstructing them candidate-by-candidate is
    fragile, so they should fall back to a full-page screenshot.
    """
    if len(candidates) < 2 or page_rect.width <= 0 or page_rect.height <= 0:
        return False

    image_masks = [
        candidate for candidate in candidates
        if candidate.get("is_image_mask") and candidate.get("bbox") is not None
    ]
    positioned_content = [
        candidate for candidate in candidates
        if not candidate.get("is_image_mask") and candidate.get("bbox") is not None
    ]
    if not image_masks or len(positioned_content) < 2:
        return False

    for candidate in image_masks:
        bbox = candidate["bbox"]
        mask_width = abs(bbox[2] - bbox[0])
        mask_height = abs(bbox[3] - bbox[1])
        if (
            mask_width / page_rect.width >= PDF_VERTICAL_SPLIT_MIN_COVERAGE
            and mask_height / page_rect.height >= PDF_VERTICAL_SPLIT_MIN_COVERAGE
        ):
            return True

    return False


def _pick_layered_full_page_base_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the best base layer for a same-bbox full-page overlay group."""
    return max(
        candidates,
        key=lambda candidate: (
            not candidate.get("is_image_mask", False),
            not candidate["has_mask"],
            candidate["pixel_area"],
            candidate["source_dpi"] or 0.0,
        ),
    )


def _pdf_xref_is_image_mask(
    doc: pymupdf.Document,
    xref: int,
    mask_flag_cache: dict[int, bool],
) -> bool:
    """Return True when the PDF image xref is an ImageMask stencil."""
    cached = mask_flag_cache.get(xref)
    if cached is not None:
        return cached

    try:
        object_text = doc.xref_object(xref, compressed=False)
    except Exception:
        is_image_mask = False
    else:
        is_image_mask = "/ImageMask true" in object_text

    mask_flag_cache[xref] = is_image_mask
    return is_image_mask


def _build_pdf_image_mask_overlay(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Build a black RGBA stencil overlay from a PDF ImageMask bitmap."""
    prepared_image = _prepare_image_for_output(image)
    try:
        if prepared_image.mode == "L":
            alpha_source = prepared_image.copy()
        else:
            alpha_source = prepared_image.convert("L")

        try:
            if alpha_source.size != target_size:
                alpha_mask = alpha_source.resize(target_size, resample=Image.LANCZOS)
            else:
                alpha_mask = alpha_source.copy()

            try:
                overlay = Image.new("RGBA", target_size, (0, 0, 0, 0))
                overlay.putalpha(alpha_mask)
                return overlay
            finally:
                alpha_mask.close()
        finally:
            alpha_source.close()
    finally:
        prepared_image.close()


def _load_pdf_mask_segment(
    doc: pymupdf.Document,
    candidate: dict[str, Any],
    union_bbox: tuple[float, float, float, float],
    target_dpi: int,
    segment_size: tuple[int, int],
    segment_offset: tuple[int, int],
    mask_cache: dict[int, Image.Image],
) -> Image.Image | None:
    """Return a mask segment for one candidate.

    If the soft mask matches the candidate image size, it is treated as a
    layer-local mask. Otherwise it is treated as a full composite mask and
    cropped to the candidate segment. That second path covers the case where a
    page is split vertically into pieces but both pieces share the same mask.
    """
    smask_xref = candidate["smask"]
    if not smask_xref:
        return None

    if smask_xref not in mask_cache:
        mask_pix = pymupdf.Pixmap(doc, smask_xref)
        mask_cache[smask_xref] = _pixmap_to_pil(mask_pix).convert("L")

    mask_image = mask_cache[smask_xref]
    if mask_image.size == (candidate["width_px"], candidate["height_px"]):
        if mask_image.size == segment_size:
            return mask_image.copy()
        return mask_image.resize(segment_size, resample=Image.LANCZOS)

    union_size = _bbox_size_at_dpi(union_bbox, target_dpi)
    if mask_image.size == union_size:
        full_mask = mask_image
    else:
        full_mask = mask_image.resize(union_size, resample=Image.LANCZOS)

    offset_x, offset_y = segment_offset
    width_px, height_px = segment_size
    return full_mask.crop((offset_x, offset_y, offset_x + width_px, offset_y + height_px))


def _extract_pdf_composite_full_width_image(
    doc: pymupdf.Document,
    candidates: list[dict[str, Any]],
    union_bbox: tuple[float, float, float, float],
    target_dpi: int,
) -> Image.Image:
    """Composite several full-width candidates onto one output canvas."""
    canvas_size = _bbox_size_at_dpi(union_bbox, target_dpi)
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    mask_cache: dict[int, Image.Image] = {}
    image_mask_flags: dict[int, bool] = {}

    for candidate in sorted(candidates, key=lambda item: item["layer_idx"]):
        candidate_bbox = candidate["bbox"]
        if candidate_bbox is None:
            continue

        pix = pymupdf.Pixmap(doc, candidate["xref"])
        raw_image = _pixmap_to_pil(pix)
        try:
            segment_size = _bbox_size_at_dpi(candidate_bbox, target_dpi)
            segment_offset = _bbox_offset_in_union(candidate_bbox, union_bbox, target_dpi)
            if _pdf_xref_is_image_mask(doc, candidate["xref"], image_mask_flags):
                layer_image = _build_pdf_image_mask_overlay(raw_image, segment_size)
            else:
                layer_image = raw_image.convert("RGBA")
                if layer_image.size != segment_size:
                    resized_layer = layer_image.resize(segment_size, resample=Image.LANCZOS)
                    layer_image.close()
                    layer_image = resized_layer

                mask_segment = _load_pdf_mask_segment(
                    doc,
                    candidate,
                    union_bbox,
                    target_dpi,
                    segment_size,
                    segment_offset,
                    mask_cache,
                )
                if mask_segment is not None:
                    layer_image.putalpha(mask_segment)

            try:
                canvas.alpha_composite(layer_image, dest=segment_offset)
            finally:
                layer_image.close()
        finally:
            raw_image.close()

    result = Image.new("RGB", canvas.size, "white")
    result.paste(canvas, mask=canvas.getchannel("A"))
    canvas.close()
    for mask in mask_cache.values():
        mask.close()
    return result


def _extract_pdf_layered_full_page_image(
    doc: pymupdf.Document,
    candidates: list[dict[str, Any]],
) -> tuple[Image.Image, float | None]:
    """Composite same-bbox full-page layers using the highest-quality base layer."""
    ordered_candidates = sorted(candidates, key=lambda item: item["layer_idx"])
    base_candidate = _pick_layered_full_page_base_candidate(ordered_candidates)
    base_pix = pymupdf.Pixmap(doc, base_candidate["xref"])
    base_image = _pixmap_to_pil(base_pix).convert("RGB")
    result = base_image.copy()
    mask_cache: dict[int, Image.Image] = {}
    image_mask_flags: dict[int, bool] = {}

    try:
        for candidate in ordered_candidates:
            if candidate["xref"] == base_candidate["xref"]:
                continue

            overlay_pix = pymupdf.Pixmap(doc, candidate["xref"])
            raw_overlay = _pixmap_to_pil(overlay_pix)
            overlay_image: Image.Image | None = None
            try:
                if _pdf_xref_is_image_mask(doc, candidate["xref"], image_mask_flags):
                    mask_overlay = _build_pdf_image_mask_overlay(raw_overlay, result.size)
                    try:
                        overlay_mask = mask_overlay.getchannel("A")
                        try:
                            black_overlay = Image.new("RGB", result.size, "black")
                            try:
                                result.paste(black_overlay, (0, 0), overlay_mask)
                            finally:
                                black_overlay.close()
                        finally:
                            overlay_mask.close()
                    finally:
                        mask_overlay.close()
                    continue

                overlay_image = raw_overlay.convert("RGB")
                if overlay_image.size != result.size:
                    resized_overlay = overlay_image.resize(result.size, resample=Image.LANCZOS)
                    overlay_image.close()
                    overlay_image = resized_overlay

                smask_xref = candidate["smask"]
                if smask_xref:
                    if smask_xref not in mask_cache:
                        mask_pix = pymupdf.Pixmap(doc, smask_xref)
                        mask_cache[smask_xref] = _pixmap_to_pil(mask_pix).convert("L")

                    mask_image = mask_cache[smask_xref]
                    if mask_image.size != result.size:
                        mask_for_overlay = mask_image.resize(result.size, resample=Image.LANCZOS)
                    else:
                        mask_for_overlay = mask_image.copy()

                    try:
                        result.paste(overlay_image, (0, 0), mask_for_overlay)
                    finally:
                        mask_for_overlay.close()
                else:
                    result.paste(overlay_image, (0, 0))
            finally:
                raw_overlay.close()
                if overlay_image is not None:
                    overlay_image.close()

        return result, base_candidate["source_dpi"]
    finally:
        base_image.close()
        for mask in mask_cache.values():
            mask.close()


def _estimate_pdf_candidate_dpi(
    width_px: int,
    height_px: int,
    bbox: tuple[float, float, float, float] | None,
    label: str,
) -> tuple[float | None, float | None, float | None]:
    """Estimate candidate DPI from pixel dimensions and placement bbox."""
    if bbox is None:
        return None, None, None

    x0, y0, x1, y1 = bbox
    width_pts = abs(x1 - x0)
    height_pts = abs(y1 - y0)

    dpi_x = clamp_dpi(width_px * 72.0 / width_pts, f"{label}:x") if width_pts > 0 else None
    dpi_y = clamp_dpi(height_px * 72.0 / height_pts, f"{label}:y") if height_pts > 0 else None

    values = [value for value in (dpi_x, dpi_y) if value is not None]
    source_dpi = sum(values) / len(values) if values else None
    return dpi_x, dpi_y, source_dpi


def _is_approx_same_width(width_pts: float | None, largest_width_pts: float) -> bool:
    """Return True when *width_pts* is approximately the same as the largest width."""
    if width_pts is None or largest_width_pts <= 0:
        return False

    width_delta = abs(width_pts - largest_width_pts)
    tolerance = max(PDF_RECT_WIDTH_ABS_TOL_PTS, largest_width_pts * PDF_RECT_WIDTH_REL_TOL)
    return width_delta <= tolerance


def _collect_pdf_page_candidates(
    page: pymupdf.Page,
    doc: pymupdf.Document,
) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
    """Collect image candidates for one PDF page and group near-full-width ones."""
    images = page.get_images(full=True)
    if not images:
        return [], 0.0, []

    mask_xrefs = {entry[1] for entry in images if entry[1] != 0}
    image_mask_flags: dict[int, bool] = {}
    candidates: list[dict[str, Any]] = []

    for layer_idx, entry in enumerate(images):
        xref = entry[0]
        smask = entry[1]
        width_px = entry[2]
        height_px = entry[3]
        bpc = entry[4]
        colorspace = entry[5]
        name = entry[7] if len(entry) > 7 else None
        filter_name = entry[8] if len(entry) > 8 else None

        raw_rects = _get_pdf_image_rects(page, xref, name)
        rects = _filter_small_rects(raw_rects)
        bbox = _union_rects(rects)
        rect_width_pts = abs(bbox[2] - bbox[0]) if bbox is not None else None
        rect_height_pts = abs(bbox[3] - bbox[1]) if bbox is not None else None
        is_image_mask = _pdf_xref_is_image_mask(doc, xref, image_mask_flags)
        dpi_x, dpi_y, source_dpi = _estimate_pdf_candidate_dpi(
            width_px,
            height_px,
            bbox,
            label=f"page_{page.number}:xref_{xref}",
        )

        candidates.append({
            "layer_idx": layer_idx,
            "xref": xref,
            "smask": smask,
            "width_px": width_px,
            "height_px": height_px,
            "pixel_area": width_px * height_px,
            "bpc": bpc,
            "colorspace": colorspace,
            "name": name,
            "filter": filter_name,
            "raw_rect_count": len(raw_rects),
            "rect_count": len(rects),
            "filtered_small_rect_count": len(raw_rects) - len(rects),
            "bbox": bbox,
            "rect_width_pts": rect_width_pts,
            "rect_height_pts": rect_height_pts,
            "dpi_x": dpi_x,
            "dpi_y": dpi_y,
            "source_dpi": source_dpi,
            "has_mask": smask != 0,
            "is_soft_mask": xref in mask_xrefs,
            "is_image_mask": is_image_mask,
            "consider_in_width_check": xref not in mask_xrefs and not is_image_mask,
            "same_width_as_largest": False,
        })

    width_candidates = sorted(
        (candidate for candidate in candidates if candidate["consider_in_width_check"]),
        key=lambda candidate: (
            candidate["rect_width_pts"] or 0.0,
            candidate["pixel_area"],
        ),
        reverse=True,
    )
    largest_rect_width_pts = width_candidates[0]["rect_width_pts"] or 0.0 if width_candidates else 0.0
    same_width_candidates: list[dict[str, Any]] = []
    for candidate in width_candidates:
        candidate["same_width_as_largest"] = _is_approx_same_width(
            candidate["rect_width_pts"],
            largest_rect_width_pts,
        )
        if candidate["same_width_as_largest"]:
            same_width_candidates.append(candidate)

    return candidates, largest_rect_width_pts, same_width_candidates


# ---------------------------------------------------------------------------
# Inspection stage
# ---------------------------------------------------------------------------

def inspect_tiff_source(source: SourceRef) -> list[PageInspection]:
    """Inspect a TIFF folder and return one page observation per TIFF file."""
    inspections: list[PageInspection] = []
    tiff_files = sorted(
        item for item in source.path.iterdir()
        if item.suffix.lower() in (".tif", ".tiff")
    )

    for page_index, tiff_path in enumerate(tiff_files):
        with Image.open(tiff_path) as image:
            dpi_w, dpi_h = get_tiff_dpi(image)
            source_dpi, warnings = summarize_source_dpi(dpi_w, dpi_h, tiff_path.name)

            inspections.append(PageInspection(
                page_index=page_index,
                output_name=f"page_{page_index}.png",
                extraction_mode="tiff_direct",
                source_dpi=source_dpi,
                source_width_px=image.width,
                source_height_px=image.height,
                payload={"tiff_path": str(tiff_path)},
                warnings=warnings,
            ))

    return inspections


def inspect_pdf_source(source: SourceRef) -> list[PageInspection]:
    """Inspect a PDF and choose one extraction mode per page.

    Preference order is: layered full-page overlays, vertical split render,
    shared-bbox composite, single widest embedded image, and page render as the
    final fallback.
    """
    inspections: list[PageInspection] = []

    with pymupdf.open(str(source.path)) as doc:
        for page_index, page in enumerate(doc):
            candidates, largest_rect_width_pts, same_width_candidates = _collect_pdf_page_candidates(page, doc)
            warnings: list[str] = []
            payload: dict[str, Any] = {
                "page_index": page_index,
                "page_size_pts": (page.rect.width, page.rect.height),
                "largest_rect_width_pts": largest_rect_width_pts or None,
                "width_match_tolerance": {
                    "relative": PDF_RECT_WIDTH_REL_TOL,
                    "absolute_points": PDF_RECT_WIDTH_ABS_TOL_PTS,
                },
                "candidates": candidates,
            }

            if not candidates:
                warnings.append("no_embedded_images")
                inspections.append(PageInspection(
                    page_index=page_index,
                    output_name=f"page_{page_index}.png",
                    extraction_mode="pdf_render",
                    payload={**payload, "decision_reason": "no_embedded_images"},
                    warnings=warnings,
                ))
                continue

            if _has_full_page_image_mask_with_other_candidates(candidates, page.rect):
                warnings.append("full_page_image_mask_screenshot")
                inspections.append(PageInspection(
                    page_index=page_index,
                    output_name=f"page_{page_index}.png",
                    extraction_mode="pdf_screenshot",
                    payload={
                        **payload,
                        "decision_reason": "full_page_image_mask_candidates",
                        "image_mask_candidate_xrefs": [
                            candidate["xref"]
                            for candidate in candidates
                            if candidate.get("is_image_mask")
                        ],
                    },
                    warnings=warnings,
                ))
                continue

            if same_width_candidates:
                if len(same_width_candidates) > 1:
                    warnings.append("multiple_full_width_candidates")

                if len(same_width_candidates) > 1 and _is_layered_full_page_group(same_width_candidates):
                    base_candidate = _pick_layered_full_page_base_candidate(same_width_candidates)
                    inspections.append(PageInspection(
                        page_index=page_index,
                        output_name=f"page_{page_index}.png",
                        extraction_mode="pdf_layered_full_page",
                        source_dpi=base_candidate["source_dpi"],
                        source_width_px=base_candidate["width_px"],
                        source_height_px=base_candidate["height_px"],
                        payload={
                            **payload,
                            "decision_reason": "layered_full_page_candidates",
                            "layered_candidate_xrefs": _ordered_candidate_xrefs(same_width_candidates),
                            "selected_xref": base_candidate["xref"],
                            "selected_bbox": base_candidate["bbox"],
                            "full_width_candidate_xrefs": [
                                candidate["xref"] for candidate in same_width_candidates
                            ],
                        },
                        warnings=warnings,
                    ))
                    continue

                if len(same_width_candidates) > 1 and _is_vertical_split_full_page_group(
                    same_width_candidates,
                    page.rect.height,
                ):
                    warnings.append("vertical_split_full_page_candidates")
                    inspections.append(PageInspection(
                        page_index=page_index,
                        output_name=f"page_{page_index}.png",
                        extraction_mode="pdf_render",
                        payload={
                            **payload,
                            "decision_reason": "vertical_split_full_page_candidates",
                            "full_width_candidate_xrefs": [
                                candidate["xref"] for candidate in same_width_candidates
                            ],
                        },
                        warnings=warnings,
                    ))
                    continue

                if len(same_width_candidates) > 1 and _composite_candidates_share_bboxes(same_width_candidates):
                    warnings.append("composite_full_width_screenshot")
                    inspections.append(PageInspection(
                        page_index=page_index,
                        output_name=f"page_{page_index}.png",
                        extraction_mode="pdf_screenshot",
                        payload={
                            **payload,
                            "decision_reason": "composite_full_width_candidates",
                            "composite_candidate_xrefs": _ordered_candidate_xrefs(same_width_candidates),
                            "full_width_candidate_xrefs": [
                                candidate["xref"] for candidate in same_width_candidates
                            ],
                        },
                        warnings=warnings,
                    ))
                    continue

                chosen = same_width_candidates[0]

                inspections.append(PageInspection(
                    page_index=page_index,
                    output_name=f"page_{page_index}.png",
                    extraction_mode="pdf_embedded",
                    source_dpi=chosen["source_dpi"],
                    source_width_px=chosen["width_px"],
                    source_height_px=chosen["height_px"],
                    payload={
                        **payload,
                        "decision_reason": "widest_candidate_after_filtering",
                        "selected_xref": chosen["xref"],
                        "selected_bbox": chosen["bbox"],
                        "selected_name": chosen["name"],
                        "full_width_candidate_xrefs": [
                            candidate["xref"] for candidate in same_width_candidates
                        ],
                    },
                    warnings=warnings,
                ))
                continue

            warnings.append("no_positioned_candidates")
            inspections.append(PageInspection(
                page_index=page_index,
                output_name=f"page_{page_index}.png",
                extraction_mode="pdf_render",
                payload={**payload, "decision_reason": "no_positioned_candidates"},
                warnings=warnings,
            ))

    return inspections


def inspect_source(source: SourceRef) -> list[PageInspection]:
    """Dispatch source inspection by source type."""
    if source.kind == "pdf":
        return inspect_pdf_source(source)
    if source.kind == "tiff_folder":
        return inspect_tiff_source(source)
    raise ValueError(f"Unsupported source type: {source.kind}")


# ---------------------------------------------------------------------------
# Extraction stage
# ---------------------------------------------------------------------------

def extract_tiff_page_image(
    inspection: PageInspection,
    config: PipelineConfig,
) -> tuple[Image.Image, float, bool]:
    """Load a TIFF page and normalize it to the target DPI."""
    tiff_path = Path(inspection.payload["tiff_path"])

    with Image.open(tiff_path) as image:
        if image.mode not in ("RGB", "L"):
            base_image = image.convert("RGB")
        else:
            base_image = image.copy()

    return normalize_to_target_dpi(
        base_image,
        inspection.source_dpi,
        config.target_dpi,
        inspection=inspection,
    )


def extract_pdf_page_image(
    source: SourceRef,
    inspection: PageInspection,
    config: PipelineConfig,
) -> tuple[Image.Image, float, bool]:
    """Build the final output image for one inspected PDF page.

    This function should consume whatever payload was assembled during
    `inspect_pdf_source` and return:
    - the final PIL image to save
    - the scale factor that was applied
    - whether rescaling happened
    """
    if inspection.extraction_mode == "pdf_embedded":
        selected_xref = inspection.payload["selected_xref"]

        with pymupdf.open(str(source.path)) as doc:
            pix = pymupdf.Pixmap(doc, selected_xref)
            base_image = _pixmap_to_pil(pix)

        try:
            return normalize_to_target_dpi(
                base_image,
                inspection.source_dpi,
                config.target_dpi,
                inspection=inspection,
            )
        finally:
            base_image.close()

    if inspection.extraction_mode == "pdf_layered_full_page":
        selected_xrefs = set(inspection.payload["layered_candidate_xrefs"])
        candidates = [
            candidate
            for candidate in inspection.payload["candidates"]
            if candidate["xref"] in selected_xrefs
        ]

        with pymupdf.open(str(source.path)) as doc:
            layered_image, source_dpi = _extract_pdf_layered_full_page_image(
                doc,
                candidates,
            )

        try:
            return normalize_to_target_dpi(
                layered_image,
                source_dpi,
                config.target_dpi,
                inspection=inspection,
            )
        finally:
            layered_image.close()

    if inspection.extraction_mode == "pdf_composite_full_width":
        composite_bbox = inspection.payload["composite_bbox"]
        selected_xrefs = set(inspection.payload["composite_candidate_xrefs"])
        candidates = [
            candidate
            for candidate in inspection.payload["candidates"]
            if candidate["xref"] in selected_xrefs
        ]

        with pymupdf.open(str(source.path)) as doc:
            composite_image = _extract_pdf_composite_full_width_image(
                doc,
                candidates,
                composite_bbox,
                config.target_dpi,
            )

        scale_factor = 1.0
        if inspection.source_dpi is not None and inspection.source_dpi > 0:
            scale_factor = config.target_dpi / inspection.source_dpi
        return composite_image, scale_factor, True

    if inspection.extraction_mode == "pdf_render":
        page_index = inspection.payload["page_index"]

        with pymupdf.open(str(source.path)) as doc:
            page = doc[page_index]
            pix = render_page_to_target_dpi(page, config.target_dpi)
            rendered_image = _pixmap_to_pil(pix)

        return rendered_image, 1.0, False

    if inspection.extraction_mode == "pdf_screenshot":
        page_index = inspection.payload["page_index"]

        with pymupdf.open(str(source.path)) as doc:
            page = doc[page_index]
            pix = page.get_pixmap(dpi=config.target_dpi, alpha=True)
            rendered_image = _pixmap_to_pil(pix)

        return rendered_image, 1.0, False

    raise ValueError(f"Unsupported PDF extraction mode: {inspection.extraction_mode}")


def extract_page_image(
    source: SourceRef,
    inspection: PageInspection,
    config: PipelineConfig,
) -> tuple[Image.Image, float, bool]:
    """Dispatch page extraction by source type."""
    if source.kind == "pdf":
        return extract_pdf_page_image(source, inspection, config)
    if source.kind == "tiff_folder":
        return extract_tiff_page_image(inspection, config)
    raise ValueError(f"Unsupported source type: {source.kind}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_source(
    source: SourceRef,
    output_root: Path,
    config: PipelineConfig,
) -> SourceResult:
    """Run the full inspect -> extract -> save flow for one source."""
    if config.skip_existing and has_existing_output(source, output_root):
        return SourceResult(source=source, status="skip", reason="existing output")

    try:
        inspections = inspect_source(source)
    except NotImplementedError as exc:
        return SourceResult(source=source, status="skip", reason=str(exc))
    except Exception as exc:
        return SourceResult(source=source, status="error", reason=str(exc))

    if not inspections:
        return SourceResult(source=source, status="skip", reason="no pages found")

    result = SourceResult(source=source, status="ok")
    out_dir = source_output_dir(source, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for inspection in inspections:
        try:
            image, scale_factor, was_rescaled = extract_page_image(source, inspection, config)
            try:
                save_page_image(image, out_dir / inspection.output_name, config.target_dpi)
            finally:
                image.close()

            page_info: dict[str, Any] = {
                "page": inspection.page_index,
                "file": inspection.output_name,
                "status": "ok",
                "extraction_mode": inspection.extraction_mode,
                "target_dpi": config.target_dpi,
                "scale_factor": round(scale_factor, 4),
                "rescaled": was_rescaled,
            }
            if inspection.source_dpi is not None:
                page_info["source_dpi"] = round(inspection.source_dpi, 2)
            if inspection.source_width_px is not None:
                page_info["source_width_px"] = inspection.source_width_px
            if inspection.source_height_px is not None:
                page_info["source_height_px"] = inspection.source_height_px
            if inspection.warnings:
                page_info["warnings"] = inspection.warnings
            if "decision_reason" in inspection.payload:
                page_info["decision_reason"] = inspection.payload["decision_reason"]
            if "dpi_correction_applied" in inspection.payload:
                page_info["dpi_correction_applied"] = inspection.payload["dpi_correction_applied"]
            if "dpi_correction_reason" in inspection.payload:
                page_info["dpi_correction_reason"] = inspection.payload["dpi_correction_reason"]
            if "reported_source_dpi" in inspection.payload:
                page_info["reported_source_dpi"] = inspection.payload["reported_source_dpi"]
            if "corrected_source_dpi" in inspection.payload:
                page_info["corrected_source_dpi"] = inspection.payload["corrected_source_dpi"]
            if "estimated_output_height_px" in inspection.payload:
                page_info["estimated_output_height_px"] = inspection.payload["estimated_output_height_px"]
            if "expected_output_height_px" in inspection.payload:
                page_info["expected_output_height_px"] = inspection.payload["expected_output_height_px"]
            if "selected_xref" in inspection.payload:
                page_info["selected_xref"] = inspection.payload["selected_xref"]
            if "full_width_candidate_xrefs" in inspection.payload:
                page_info["full_width_candidate_xrefs"] = inspection.payload["full_width_candidate_xrefs"]
            if "layered_candidate_xrefs" in inspection.payload:
                page_info["layered_candidate_xrefs"] = inspection.payload["layered_candidate_xrefs"]
            if "composite_candidate_xrefs" in inspection.payload:
                page_info["composite_candidate_xrefs"] = inspection.payload["composite_candidate_xrefs"]

            result.pages.append(page_info)
        except NotImplementedError as exc:
            result.status = "skip"
            result.reason = str(exc)
            result.pages.clear()
            return result
        except Exception as exc:
            error_info = {
                "page": inspection.page_index,
                "file": inspection.output_name,
                "status": "error",
                "error": str(exc),
            }
            result.pages.append(error_info)
            result.warnings.append(f"page_{inspection.page_index}: {exc}")

    if not any(page.get("status") == "ok" for page in result.pages):
        result.status = "error"
        result.reason = result.reason or "no pages were written"

    return result


def build_summary(results: list[SourceResult]) -> dict[str, int]:
    """Compute a small run summary for the step report."""
    return {
        "total_sources": len(results),
        "succeeded": sum(1 for result in results if result.status == "ok"),
        "skipped": sum(1 for result in results if result.status == "skip"),
        "failed": sum(1 for result in results if result.status == "error"),
        "pages_written": sum(
            1
            for result in results
            for page in result.pages
            if page.get("status") == "ok"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect PDF/TIFF sources once and extract normalized page PNGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/extract_sources.py data/corpus-1/pdfs data/corpus-1/imgs
  python scripts/extract_sources.py data/corpus-1/pdfs data/corpus-1/imgs --skip-existing
  python scripts/extract_sources.py data/corpus-1/pdfs data/corpus-1/imgs --target-dpi 300
        """,
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing PDFs and/or TIFF folders",
    )
    parser.add_argument(
        "img_dir",
        help="Destination directory for extracted PNG page images",
    )
    parser.add_argument(
        "--target-dpi",
        type=int,
        default=150,
        help="Target DPI for output images (default: 150)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sources whose output folder already contains PNGs",
    )
    parser.add_argument(
        "--report-dir",
        help="Directory for the step report JSON (default: {img_dir}/../reports)",
    )
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir).resolve()
    img_dir = Path(args.img_dir).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else img_dir.parent / "reports"

    if not source_dir.is_dir():
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        return 1

    sources = discover_sources(source_dir)
    if not sources:
        print(f"No PDF files or TIFF folders found in {source_dir}")
        return 0

    config = PipelineConfig(
        target_dpi=args.target_dpi,
        skip_existing=args.skip_existing,
    )

    n_pdfs = sum(1 for source in sources if source.kind == "pdf")
    n_tiffs = sum(1 for source in sources if source.kind == "tiff_folder")

    print(f"Sources to process: {len(sources)} ({n_pdfs} PDFs, {n_tiffs} TIFF folders)")
    print(f"Output: {img_dir}")
    print(f"Target DPI: {config.target_dpi}")

    img_dir.mkdir(parents=True, exist_ok=True)
    report = StepReport("extract_sources", target_dpi=config.target_dpi)

    start = time.time()
    results: list[SourceResult] = []
    for source in tqdm(sources, desc="Extracting sources"):
        result = process_source(source, img_dir, config)
        results.append(result)
        report.add_document(source.name, result.to_report_dict())

    report.set_summary(build_summary(results))
    report_path = report.save(report_dir)
    elapsed = time.time() - start

    summary = build_summary(results)
    print(f"Report saved: {report_path}")
    print(
        f"Done: {summary['succeeded']} succeeded, "
        f"{summary['failed']} failed, {summary['skipped']} skipped "
        f"in {elapsed:.1f}s"
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())