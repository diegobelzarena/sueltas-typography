#!/usr/bin/env python
"""Convert a directory of scanned PDFs and/or TIFF folders to normalized PNGs.

Each source (a ``.pdf`` file or a folder of ``.tif``/``.tiff`` pages) produces
a subfolder of PNGs (one per page), scaled to a target DPI (default 150).

The original DPI is determined from:
  1. A precomputed CSV file in ``--dpi-csv-dir`` (preferred), or
  2. Embedded metadata (PDF image resolution / TIFF DPI tag).

The input directory may contain a mix of PDFs and TIFF folders — both are
auto-detected and processed together.

Usage
-----
    # Convert everything in one directory (PDFs + TIFF folders)
    python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs

    # With precomputed DPI values and parallel processing
    python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs \\
        --dpi-csv-dir data/corpus-1/dpis --workers 8 --skip-existing

    # Custom target DPI
    python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs --target-dpi 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from shared.tools.report import StepReport
from image_processing.pdf_utils import (
    clamp_dpi,
    get_tiff_dpi,
    median_dpi,
    round_dpi,
    select_best_image,
)


# ---------------------------------------------------------------------------
# Special-case handlers
# ---------------------------------------------------------------------------
# Some PDFs have non-standard internal structure (e.g. layered scans with
# separate text/background images and masks).  Register them here so that
# the main extraction path stays clean.

SPECIAL_CASES: dict[str, Callable] = {}


def _register_special(name: str):
    """Decorator to register a special-case handler by PDF stem name."""
    def decorator(func):
        SPECIAL_CASES[name] = func
        return func
    return decorator


@_register_special("BNE_796_587_T-55352-9")
def _extract_layered(pdf_path: Path, out_dir: Path, target_dpi: int) -> int:
    """BNE_796_587_T-55352-9: background + text layer with smask.

    Pages have two images: a background layer and a foreground text layer
    with an alpha mask.  They are composited and saved as a single PNG.
    """
    doc = pymupdf.open(str(pdf_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Estimate DPI for scaling
    orig_dpi = _get_dpi_from_pdf(doc)
    scale = target_dpi / orig_dpi if orig_dpi else 1.0
    count = 0

    for idx, page in enumerate(doc):
        images = page.get_images()
        if not images:
            continue

        # Background image (first)
        pix = pymupdf.Pixmap(doc, images[0][0])
        h, w, c = pix.height, pix.width, pix.n
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, c)).copy()

        # Text layer (second) with mask
        if len(images) > 1:
            xref = images[1][0]
            smask = images[1][1]
            pix2 = pymupdf.Pixmap(doc, xref)
            h1, w1 = pix2.height, pix2.width
            fg_pil = Image.frombytes("RGB", (w1, h1), pix2.samples)
            fg = np.array(fg_pil.resize((w, h)))

            pixm = pymupdf.Pixmap(doc, smask)
            if (h, w) != (pixm.height, pixm.width):
                print(f"  Warning: page {idx} mask size mismatch, skipping composite")
            else:
                mask = np.frombuffer(pixm.samples, dtype=np.bool_).reshape((h, w))
                img[mask] = fg[mask]

        result = Image.fromarray(img)
        if abs(scale - 1.0) > 0.01:
            new_w = int(result.width * scale)
            new_h = int(result.height * scale)
            result = result.resize((new_w, new_h), resample=Image.LANCZOS)
        result.save(str(out_dir / f"page_{idx}.png"), dpi=(target_dpi, target_dpi))
        count += 1

    doc.close()
    return count


# ---------------------------------------------------------------------------
# DPI helpers (shared by PDF and TIFF paths)
# ---------------------------------------------------------------------------

def _get_dpi_from_csv(name: str, dpi_csv_dir: Path) -> float | None:
    """Read the median DPI from a precomputed CSV, if available.

    Works for both PDF stems and TIFF folder names.
    """
    csv_path = dpi_csv_dir / (name + ".csv")
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        values: list[float] = []
        for col in ("DPI (width)", "DPI (height)"):
            if col in df.columns:
                values.extend(df[col].dropna().tolist())
        if values:
            return median_dpi(values, step=25)
    except Exception as e:
        print(f"  Warning: failed to read DPI csv {csv_path.name}: {e}")

    return None


def _get_dpi_from_pdf(doc: pymupdf.Document) -> float:
    """Estimate DPI from embedded image metadata in the PDF."""
    dpi_values: list[float] = []
    for page in doc:
        best_xref, _, _, _ = select_best_image(page, doc)
        if best_xref == 0:
            continue
        pix = pymupdf.Pixmap(doc, best_xref)
        if pix.xres:
            dpi_values.append(clamp_dpi(pix.xres))
        elif pix.yres:
            dpi_values.append(clamp_dpi(pix.yres))

    return median_dpi(dpi_values, step=25)


def _get_dpi_from_tiffs(tiff_files: list[Path]) -> float:
    """Estimate DPI from TIFF metadata across a set of page files.

    Handles resolution-unit (tag 296) correctly via :func:`get_tiff_dpi`.
    """
    dpi_values: list[float] = []
    for tf in tiff_files:
        try:
            with Image.open(tf) as img:
                dpi_w, dpi_h = get_tiff_dpi(img)
                if dpi_w is not None:
                    dpi_values.append(clamp_dpi(dpi_w, tf.name))
                if dpi_h is not None:
                    dpi_values.append(clamp_dpi(dpi_h, tf.name))
        except Exception:
            continue

    return median_dpi(dpi_values, step=25)


# ---------------------------------------------------------------------------
# Core extraction: PDF
# ---------------------------------------------------------------------------

def _pixmap_to_pil(pix: pymupdf.Pixmap) -> Image.Image:
    """Convert a PyMuPDF Pixmap to a Pillow Image."""
    # CMYK pixmaps: convert to RGB first to avoid misinterpreting channels
    if pix.colorspace and pix.colorspace.n == 4 and not pix.alpha:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    mode = "RGB" if pix.n >= 3 else "L"
    if pix.alpha:
        mode += "A"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def extract_pdf(
    pdf_path: Path,
    out_dir: Path,
    target_dpi: int = 150,
    dpi_csv_dir: Path | None = None,
) -> tuple[str, dict]:
    """Extract page images from a single PDF.

    Returns (status_string, info_dict) for reporting.
    """
    pdf_stem = pdf_path.stem
    page_dir = out_dir / pdf_stem
    info: dict = {"source_name": pdf_stem, "source_type": "pdf"}

    # Check for special-case handler
    if pdf_stem in SPECIAL_CASES:
        handler = SPECIAL_CASES[pdf_stem]
        count = handler(pdf_path, page_dir, target_dpi)
        info.update(special_handling=True, n_pages=count, status="ok")
        return f"OK (special): {pdf_stem} — {count} pages", info

    # Standard extraction
    doc = pymupdf.open(str(pdf_path))
    page_dir.mkdir(parents=True, exist_ok=True)

    # Determine original DPI
    orig_dpi: float | None = None
    dpi_source = "fallback"
    if dpi_csv_dir:
        orig_dpi = _get_dpi_from_csv(pdf_stem, dpi_csv_dir)
        if orig_dpi is not None:
            dpi_source = "csv"
    if orig_dpi is None:
        orig_dpi = _get_dpi_from_pdf(doc)
        if orig_dpi != 150:
            dpi_source = "pdf_metadata"

    scale = target_dpi / orig_dpi if orig_dpi else 1.0
    count = 0
    page_details: list[dict] = []
    warnings: list[str] = []

    for idx, page in enumerate(doc):
        images = page.get_images(full=True)
        n_layers = len(images)
        if not images:
            page_details.append({"page": idx, "n_layers": 0, "status": "skip_no_images"})
            continue

        best_xref, _best_w, _best_h, mask_xrefs = select_best_image(page, doc)
        if best_xref == 0:
            page_details.append({"page": idx, "n_layers": n_layers, "status": "skip_no_valid_image"})
            continue

        if n_layers != 1:
            msg = (f"{pdf_stem} page {idx} has {n_layers} images "
                   f"({len(mask_xrefs)} mask(s)), selected xref {best_xref}")
            warnings.append(msg)
            print(f"  Info: {msg}")

        pix = pymupdf.Pixmap(doc, best_xref)

        if abs(scale - 1.0) > 0.01:
            img = _pixmap_to_pil(pix)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), resample=Image.LANCZOS)
            img.save(str(page_dir / f"page_{idx}.png"), dpi=(target_dpi, target_dpi))
            pw, ph = new_w, new_h
        else:
            pix.save(str(page_dir / f"page_{idx}.png"))
            pw, ph = pix.width, pix.height

        page_details.append({
            "page": idx, "n_layers": n_layers,
            "width_px": pw, "height_px": ph, "status": "ok",
        })
        count += 1

    doc.close()
    info.update(
        n_pages=count,
        original_dpi=orig_dpi,
        target_dpi=target_dpi,
        scale_factor=round(scale, 4),
        dpi_source=dpi_source,
        special_handling=False,
        pages=page_details,
        status="ok",
    )
    if warnings:
        info["warnings"] = warnings
    return (
        f"OK: {pdf_stem} — {count} pages [PDF] "
        f"(DPI {orig_dpi}→{target_dpi}, scale {scale:.2f})",
        info,
    )


# ---------------------------------------------------------------------------
# Core extraction: TIFF folder
# ---------------------------------------------------------------------------

def extract_tiff_folder(
    tiff_dir: Path,
    out_dir: Path,
    target_dpi: int = 150,
    dpi_csv_dir: Path | None = None,
) -> tuple[str, dict]:
    """Convert a folder of TIFF pages to PNGs at a target DPI.

    Returns (status_string, info_dict) for reporting.
    """
    folder_name = tiff_dir.name
    page_dir = out_dir / folder_name
    info: dict = {"source_name": folder_name, "source_type": "tiff_folder"}

    tiff_files = sorted(
        f for f in tiff_dir.iterdir()
        if f.suffix.lower() in (".tif", ".tiff")
    )
    if not tiff_files:
        info.update(n_pages=0, status="skip", reason="no TIFF files")
        return f"SKIP: {folder_name} — no TIFF files", info

    page_dir.mkdir(parents=True, exist_ok=True)

    # Determine original DPI
    orig_dpi: float | None = None
    dpi_source = "fallback"
    if dpi_csv_dir:
        orig_dpi = _get_dpi_from_csv(folder_name, dpi_csv_dir)
        if orig_dpi is not None:
            dpi_source = "csv"
    if orig_dpi is None:
        orig_dpi = _get_dpi_from_tiffs(tiff_files)
        if orig_dpi != 150:
            dpi_source = "tiff_metadata"

    scale = target_dpi / orig_dpi if orig_dpi else 1.0
    count = 0
    page_details: list[dict] = []
    warnings: list[str] = []

    for idx, tiff_path in enumerate(tiff_files):
        try:
            with Image.open(tiff_path) as img:
                # Convert to RGB if needed (handles palette, CMYK, etc.)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                if abs(scale - 1.0) > 0.01:
                    new_w = int(img.width * scale)
                    new_h = int(img.height * scale)
                    img = img.resize((new_w, new_h), resample=Image.LANCZOS)
                else:
                    new_w, new_h = img.width, img.height

                img.save(
                    str(page_dir / f"page_{idx}.png"),
                    dpi=(target_dpi, target_dpi),
                )
                page_details.append({
                    "page": idx, "width_px": new_w, "height_px": new_h,
                    "status": "ok",
                })
                count += 1
        except Exception as e:
            msg = f"{folder_name}/{tiff_path.name}: {e}"
            warnings.append(msg)
            print(f"  Warning: {msg}")
            page_details.append({"page": idx, "status": "error", "error": str(e)})

    info.update(
        n_pages=count,
        original_dpi=orig_dpi,
        target_dpi=target_dpi,
        scale_factor=round(scale, 4),
        dpi_source=dpi_source,
        pages=page_details,
        status="ok",
    )
    if warnings:
        info["warnings"] = warnings
    return (
        f"OK: {folder_name} — {count} pages [TIFF] "
        f"(DPI {orig_dpi}→{target_dpi}, scale {scale:.2f})",
        info,
    )


# ---------------------------------------------------------------------------
# Unified task dispatcher
# ---------------------------------------------------------------------------

def _convert_one(source: Path, out_dir: Path, target_dpi: int,
                 dpi_csv_dir: Path | None) -> tuple[str, dict]:
    """Convert a single source (PDF file or TIFF folder) to PNGs."""
    if source.is_file() and source.suffix.lower() == ".pdf":
        return extract_pdf(source, out_dir, target_dpi, dpi_csv_dir)
    elif source.is_dir():
        return extract_tiff_folder(source, out_dir, target_dpi, dpi_csv_dir)
    else:
        info = {"source_name": source.name, "source_type": "unknown", "status": "skip"}
        return f"SKIP: {source.name} — unsupported type", info


def _convert_wrapper(args: tuple) -> tuple[str, dict]:
    """Picklable wrapper for ProcessPoolExecutor."""
    return _convert_one(*args)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _is_tiff_folder(path: Path) -> bool:
    """Check if a directory contains TIFF images."""
    return path.is_dir() and any(
        f.suffix.lower() in (".tif", ".tiff") for f in path.iterdir()
    )


def discover_sources(root: Path) -> list[Path]:
    """Find all convertible sources (PDFs and TIFF folders) under *root*.

    Returns a sorted list of PDF files and directories containing TIFFs.
    """
    sources: list[Path] = []
    for item in sorted(root.iterdir()):
        if item.is_file() and item.suffix.lower() == ".pdf":
            sources.append(item)
        elif _is_tiff_folder(item):
            sources.append(item)
    return sources


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert scanned PDFs and TIFF folders to normalized PNG page images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs
  python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs \\
      --dpi-csv-dir data/corpus-1/dpis --workers 8 --skip-existing
  python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs --target-dpi 300
        """,
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing PDF files and/or TIFF folders",
    )
    parser.add_argument(
        "img_dir",
        help="Destination directory for output image folders",
    )
    parser.add_argument(
        "--dpi-csv-dir",
        help="Directory with precomputed per-source DPI CSVs (from compute_dpi.py)",
        default=None,
    )
    parser.add_argument(
        "--target-dpi",
        type=int,
        default=150,
        help="Target DPI for output images (default: 150)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers (default: ncpus-1, 0 = auto)",
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
    dpi_csv_dir = Path(args.dpi_csv_dir).resolve() if args.dpi_csv_dir else None
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    # Validate
    if not source_dir.is_dir():
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        return 1

    # Discover sources
    sources = discover_sources(source_dir)
    if not sources:
        print(f"No PDF files or TIFF folders found in {source_dir}")
        return 0

    n_pdfs = sum(1 for s in sources if s.is_file())
    n_tiffs = sum(1 for s in sources if s.is_dir())

    # Filter skip-existing
    if args.skip_existing:
        original = len(sources)

        def _source_name(s: Path) -> str:
            return s.stem if s.is_file() else s.name

        sources = [
            s for s in sources
            if not (img_dir / _source_name(s)).is_dir()
            or not any((img_dir / _source_name(s)).glob("*.png"))
        ]
        skipped = original - len(sources)
        if skipped:
            print(f"Skipping {skipped} sources with existing images")
        n_pdfs = sum(1 for s in sources if s.is_file())
        n_tiffs = sum(1 for s in sources if s.is_dir())

    print(f"Sources to convert: {len(sources)} ({n_pdfs} PDFs, {n_tiffs} TIFF folders)")
    print(f"Output: {img_dir}")
    print(f"Target DPI: {args.target_dpi}")
    if dpi_csv_dir:
        print(f"DPI CSVs: {dpi_csv_dir}")

    img_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Build tasks
    tasks = [(s, img_dir, args.target_dpi, dpi_csv_dir) for s in sources]
    report = StepReport("convert_sources", target_dpi=args.target_dpi)

    if workers > 1 and len(tasks) > 1:
        print(f"Workers: {workers}")
        results: list[str] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_convert_wrapper, t): (t[0].stem if t[0].is_file() else t[0].name)
                for t in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Converting"):
                try:
                    msg, info = future.result()
                    results.append(msg)
                    report.add_document(info["source_name"], info)
                except Exception as e:
                    name = futures[future]
                    results.append(f"FAIL: {name} — {e}")
                    report.add_document(name, {"status": "error", "error": str(e)})
    else:
        results = []
        for task in tqdm(tasks, desc="Converting"):
            try:
                msg, info = _convert_wrapper(task)
                results.append(msg)
                report.add_document(info["source_name"], info)
            except Exception as e:
                name = task[0].stem if task[0].is_file() else task[0].name
                results.append(f"FAIL: {name} — {e}")
                report.add_document(name, {"status": "error", "error": str(e)})

    # Summary
    ok = sum(1 for r in results if r.startswith("OK"))
    fail = sum(1 for r in results if r.startswith("FAIL"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    elapsed = time.time() - start

    total_pages = sum(
        d.get("n_pages", 0)
        for d in report.documents.values()
    )
    report.set_summary({
        "total_sources": len(tasks),
        "succeeded": ok,
        "failed": fail,
        "skipped": skip,
        "total_pages_extracted": total_pages,
    })

    # Save report next to the output images
    report_dir = Path(args.report_dir) if args.report_dir else img_dir.parent / "reports"
    report_path = report.save(report_dir)
    print(f"Report saved: {report_path}")

    print(f"\nDone: {ok} succeeded, {fail} failed, {skip} skipped in {elapsed:.1f}s")
    for r in results:
        if r.startswith("FAIL"):
            print(f"  {r}")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
