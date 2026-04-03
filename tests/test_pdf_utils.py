"""Unit tests for :mod:`image_processing.pdf_utils`.

All tests use **synthetic image-info dictionaries** and/or mocked PyMuPDF
objects — no real PDF files are required.

Coverage
--------
* :func:`_rank_scan_candidates` — selection heuristic (pure-Python, no mocks)
* :func:`estimate_effective_dpi` — DPI formula, clamping, fallback paths
* :func:`clamp_dpi` / :func:`round_dpi` — regression tests for existing helpers
* :func:`select_best_scan_image` — integration test via mocked pymupdf.Page
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from image_processing.pdf_utils import (
    MIN_COVERAGE,
    MIN_PIXEL_AREA,
    DPI_MIN,
    DPI_MAX,
    ScanImageResult,
    _rank_scan_candidates,
    clamp_dpi,
    estimate_effective_dpi,
    round_dpi,
    select_best_scan_image,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic candidates
# ---------------------------------------------------------------------------

def make_candidate(
    xref: int,
    width: int,
    height: int,
    coverage: float,
    bbox: tuple[float, float, float, float] | None = None,
    name: str | None = None,
) -> dict:
    """Build a synthetic image-info dict (no real PDF needed)."""
    if bbox is None:
        bbox = (0.0, 0.0, float(width), float(height))
    return {
        "xref": xref,
        "width": width,
        "height": height,
        "pixel_area": width * height,
        "coverage": coverage,
        "bbox": bbox,
        "name": name or str(xref),
    }


def make_scan_result(
    xref: int,
    width: int,
    height: int,
    coverage: float = 0.95,
    bbox: tuple[float, float, float, float] | None = None,
) -> ScanImageResult:
    """Build a :class:`ScanImageResult` TypedDict for DPI tests."""
    if bbox is None:
        bbox = (0.0, 0.0, float(width), float(height))
    return ScanImageResult(
        xref=xref,
        width=width,
        height=height,
        pixel_area=width * height,
        coverage=coverage,
        bbox=bbox,
    )


def make_page_mock(width_pts: float = 595.0, height_pts: float = 842.0) -> MagicMock:
    """Return a minimal mock for a :class:`pymupdf.Page`."""
    page = MagicMock()
    page.rect = MagicMock()
    page.rect.width = width_pts
    page.rect.height = height_pts
    return page


# ===========================================================================
# _rank_scan_candidates
# ===========================================================================

class TestRankScanCandidates:
    """Selection heuristic — all tests are pure Python, zero I/O."""

    # --- basic filtering ---

    def test_single_high_coverage_image_is_returned(self):
        cands = [make_candidate(1, 2000, 3000, 0.95)]
        ranked = _rank_scan_candidates(cands)
        assert len(ranked) == 1
        assert ranked[0]["xref"] == 1

    def test_thumbnail_loses_to_lower_resolution_full_page(self):
        """Thumbnail has more pixels but lower coverage — full-page wins."""
        full_page = make_candidate(1, 2000, 3000, 0.95)
        thumbnail = make_candidate(2, 5000, 7000, 0.08)  # many pixels, tiny bbox
        ranked = _rank_scan_candidates([thumbnail, full_page])
        assert ranked[0]["xref"] == 1, "Full-page image must win despite fewer pixels"

    def test_off_page_image_ranked_last(self):
        """Image placed outside page area has near-zero coverage → filtered."""
        full_page = make_candidate(1, 2000, 3000, 0.90)
        off_page = make_candidate(2, 5000, 7000, 0.01)
        ranked = _rank_scan_candidates([off_page, full_page])
        assert ranked[0]["xref"] == 1

    def test_icon_rejected_by_pixel_area(self):
        """Watermark-sized image (tiny pixel_area) filtered even if coverage is OK."""
        icon = make_candidate(1, 80, 80, 0.95)      # pixel_area = 6400 < 10000
        scan = make_candidate(2, 2000, 3000, 0.90)
        ranked = _rank_scan_candidates([icon, scan])
        assert ranked[0]["xref"] == 2

    # --- ranking / tiebreaking ---

    def test_higher_coverage_wins(self):
        a = make_candidate(1, 2000, 3000, 0.80)
        b = make_candidate(2, 2000, 3000, 0.95)
        ranked = _rank_scan_candidates([a, b])
        assert ranked[0]["xref"] == 2

    def test_pixel_area_breaks_coverage_tie(self):
        """Equal coverage → larger image wins."""
        a = make_candidate(1, 2000, 3000, 0.92)  # pixel_area = 6_000_000
        b = make_candidate(2, 3000, 4000, 0.92)  # pixel_area = 12_000_000
        ranked = _rank_scan_candidates([a, b])
        assert ranked[0]["xref"] == 2

    # --- fallback paths ---

    def test_fallback_when_no_candidate_meets_both_thresholds(self):
        """All images below MIN_COVERAGE → coverage-only fallback, sorted."""
        a = make_candidate(1, 2000, 3000, 0.30)
        b = make_candidate(2, 1000, 1500, 0.20)
        ranked = _rank_scan_candidates([a, b])
        assert len(ranked) == 2
        assert ranked[0]["xref"] == 1  # higher coverage

    def test_fallback_when_pixel_area_fails_but_coverage_passes(self):
        """Pixel area below MIN_PIXEL_AREA → relaxed fallback includes image."""
        small_full = make_candidate(1, 80, 80, 0.95)  # coverage OK, pixels not
        ranked = _rank_scan_candidates([small_full])
        assert len(ranked) == 1
        assert ranked[0]["xref"] == 1

    def test_unconditional_fallback_when_all_fail(self):
        """All images fail both thresholds → return all, sorted."""
        a = make_candidate(1, 50, 50, 0.10)
        b = make_candidate(2, 60, 60, 0.05)
        ranked = _rank_scan_candidates([a, b])
        assert len(ranked) == 2
        assert ranked[0]["xref"] == 1  # higher coverage

    def test_empty_candidates_returns_empty(self):
        assert _rank_scan_candidates([]) == []

    def test_returns_all_qualified_sorted(self):
        a = make_candidate(1, 2000, 3000, 0.90)
        b = make_candidate(2, 2500, 3500, 0.95)
        c = make_candidate(3, 100, 100, 0.05)  # below both thresholds
        ranked = _rank_scan_candidates([a, b, c])
        # c should end up last (or excluded from top)
        assert ranked[0]["xref"] == 2
        assert ranked[1]["xref"] == 1

    def test_custom_thresholds_respected(self):
        """Custom min_coverage=0.8 should exclude images at 0.75."""
        a = make_candidate(1, 2000, 3000, 0.75)  # would pass default 0.5
        b = make_candidate(2, 2000, 3000, 0.90)
        ranked = _rank_scan_candidates([a, b], min_coverage=0.80)
        assert ranked[0]["xref"] == 2


# ===========================================================================
# estimate_effective_dpi
# ===========================================================================

class TestEstimateEffectiveDpi:
    """DPI formula: dpi = pixels / pts * 72."""

    def test_300dpi_a4(self):
        """2480 × 3508 px on 595 × 842 pt page → ~300 DPI."""
        result = make_scan_result(1, 2480, 3508, bbox=(0, 0, 595, 842))
        page = make_page_mock(595, 842)
        dpi_x, dpi_y = estimate_effective_dpi(result, page)
        assert dpi_x is not None
        assert 290 <= dpi_x <= 310, f"Expected ~300, got {dpi_x}"

    def test_150dpi_a4(self):
        """1240 × 1754 px → ~150 DPI."""
        result = make_scan_result(1, 1240, 1754, bbox=(0, 0, 595, 842))
        page = make_page_mock(595, 842)
        dpi_x, dpi_y = estimate_effective_dpi(result, page)
        assert dpi_x is not None
        assert 140 <= dpi_x <= 160, f"Expected ~150, got {dpi_x}"

    def test_fallback_to_page_rect_when_no_bbox(self):
        """No bbox supplied → falls back to page.rect dimensions."""
        result = make_scan_result(1, 2000, 3000, bbox=None)
        page = make_page_mock(595, 842)
        dpi_x, dpi_y = estimate_effective_dpi(result, page)
        assert dpi_x is not None and dpi_x > 0

    def test_result_clamped_to_dpi_max(self):
        """Absurdly large pixel count → clamped to DPI_MAX."""
        result = make_scan_result(1, 999_999, 999_999, bbox=(0, 0, 595, 842))
        page = make_page_mock(595, 842)
        dpi_x, _ = estimate_effective_dpi(result, page)
        assert dpi_x is not None
        assert dpi_x == float(DPI_MAX)

    def test_zero_width_bbox_returns_none_for_x(self):
        """Degenerate bbox with zero width → dpi_x is None."""
        result = make_scan_result(1, 2000, 3000, bbox=(100, 0, 100, 842))
        page = make_page_mock()
        dpi_x, dpi_y = estimate_effective_dpi(result, page)
        assert dpi_x is None
        assert dpi_y is not None  # height dimension is fine

    def test_zero_height_bbox_returns_none_for_y(self):
        """Degenerate bbox with zero height → dpi_y is None."""
        result = make_scan_result(1, 2000, 3000, bbox=(0, 100, 595, 100))
        page = make_page_mock()
        dpi_x, dpi_y = estimate_effective_dpi(result, page)
        assert dpi_y is None
        assert dpi_x is not None

    def test_both_axes_independent(self):
        """Horizontal and vertical DPI can differ (non-square pixels)."""
        # 2000 px wide on 500 pts → 2000/500*72 = 288 dpi_x
        # 3000 px tall on 1000 pts → 3000/1000*72 = 216 dpi_y
        result = make_scan_result(1, 2000, 3000, bbox=(0, 0, 500, 1000))
        page = make_page_mock()
        dpi_x, dpi_y = estimate_effective_dpi(result, page)
        assert dpi_x is not None and dpi_y is not None
        assert abs(dpi_x - 288) < 5
        assert abs(dpi_y - 216) < 5


# ===========================================================================
# clamp_dpi / round_dpi — regression tests for existing helpers
# ===========================================================================

class TestDpiHelpers:
    def test_clamp_below_min(self):
        assert clamp_dpi(10.0) == float(DPI_MIN)

    def test_clamp_above_max(self):
        assert clamp_dpi(9999.0) == float(DPI_MAX)

    def test_clamp_passthrough_middle(self):
        assert clamp_dpi(300.0) == 300.0

    def test_clamp_at_boundary_min(self):
        assert clamp_dpi(float(DPI_MIN)) == float(DPI_MIN)

    def test_clamp_at_boundary_max(self):
        assert clamp_dpi(float(DPI_MAX)) == float(DPI_MAX)

    def test_round_to_nearest_25_down(self):
        assert round_dpi(311) == 300

    def test_round_to_nearest_25_up(self):
        assert round_dpi(313) == 325

    def test_round_minimum_floor(self):
        assert round_dpi(10) == DPI_MIN

    def test_round_exact_multiple(self):
        assert round_dpi(200) == 200


# ===========================================================================
# select_best_scan_image — integration test via mocked pymupdf.Page
# ===========================================================================

class TestSelectBestScanImage:
    """Integration tests that mock pymupdf.Page / Document.

    The mock simulates ``page.get_images(full=True)`` and
    ``page.get_image_rects(name)`` to verify the full selection pipeline.
    """

    def _make_mock_page(
        self,
        images: list[tuple],
        rects_map: dict[str, list] | None = None,
        width_pts: float = 595.0,
        height_pts: float = 842.0,
    ) -> MagicMock:
        """Build a mock pymupdf.Page.

        Parameters
        ----------
        images : list[tuple]
            Each tuple is (xref, smask, width, height, bpc, cs, alt_cs, name).
        rects_map : dict[str, list[pymupdf.Rect]]
            Maps image name → list of placement rects.  If omitted, each
            image gets a rect covering the full page.
        """
        import pymupdf

        page = MagicMock()
        page.rect = pymupdf.Rect(0, 0, width_pts, height_pts)
        page.get_images.return_value = images

        full_rect = [pymupdf.Rect(0, 0, width_pts, height_pts)]

        def get_image_rects(name):
            if rects_map and name in rects_map:
                return rects_map[name]
            return full_rect

        page.get_image_rects.side_effect = get_image_rects
        return page

    def test_returns_none_when_no_images(self):
        page = self._make_mock_page(images=[])
        doc = MagicMock()
        result = select_best_scan_image(page, doc)
        assert result is None

    def test_single_image_always_selected(self):
        images = [(1, 0, 2000, 3000, 8, "DeviceGray", "", "img1", "FlateDecode", 0)]
        page = self._make_mock_page(images)
        result = select_best_scan_image(page, MagicMock())
        assert result is not None
        assert result["xref"] == 1

    def test_mask_xref_excluded(self):
        """smask image (xref=2) should be excluded; only xref=1 is a candidate."""
        images = [
            (1, 2, 2000, 3000, 8, "DeviceRGB", "", "img1", "FlateDecode", 0),
            (2, 0, 2000, 3000, 8, "DeviceGray", "", "mask1", "FlateDecode", 0),
        ]
        page = self._make_mock_page(images)
        result = select_best_scan_image(page, MagicMock())
        assert result is not None
        assert result["xref"] == 1

    def test_thumbnail_loses_due_to_small_placement(self):
        """Thumbnail placed at 5% of page should lose to full-page image."""
        import pymupdf

        w, h = 595.0, 842.0
        images = [
            (1, 0, 2000, 3000, 8, "DeviceRGB", "", "scan", "FlateDecode", 0),
            (2, 0, 4000, 6000, 8, "DeviceRGB", "", "thumb", "FlateDecode", 0),
        ]
        small_rect = [pymupdf.Rect(0, 0, w * 0.05, h * 0.05)]  # 5% coverage
        full_rect = [pymupdf.Rect(0, 0, w, h)]
        rects = {"scan": full_rect, "thumb": small_rect}
        page = self._make_mock_page(images, rects_map=rects)
        result = select_best_scan_image(page, MagicMock())
        assert result is not None
        assert result["xref"] == 1, "Scan image (full coverage) must win"

    def test_result_fields_are_populated(self):
        images = [(1, 0, 2000, 3000, 8, "DeviceRGB", "", "img1", "FlateDecode", 0)]
        page = self._make_mock_page(images)
        result = select_best_scan_image(page, MagicMock())
        assert result is not None
        assert result["width"] == 2000
        assert result["height"] == 3000
        assert result["pixel_area"] == 6_000_000
        assert 0.0 <= result["coverage"] <= 1.0
