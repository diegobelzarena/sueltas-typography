#!/usr/bin/env python
"""Convert a directory of scanned PDFs to normalized PNG page images.

Each PDF produces a subfolder of PNGs (one per page), scaled to a target
DPI (default 150).  The original DPI is determined from:
  1. A precomputed CSV file in ``--dpi-csv-dir`` (preferred), or
  2. The embedded image resolution metadata in the PDF.

Usage
-----
    # Basic conversion
    python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs

    # With precomputed DPI values
    python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs \\
        --dpi-csv-dir data/corpus-1/dpis

    # Parallel with 8 workers
    python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs \\
        --workers 8 --skip-existing
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Special-case handlers
from collections.abc import Callable

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

        Image.fromarray(img).save(str(out_dir / f"page_{idx}.png"))
        count += 1

    doc.close()
    return count


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _pixmap_to_pil(pix: pymupdf.Pixmap) -> Image.Image:
    """Convert a PyMuPDF Pixmap to a Pillow Image."""
    mode = "RGB" if pix.n >= 3 else "L"
    if pix.alpha:
        mode += "A"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def _get_dpi_from_csv(pdf_stem: str, dpi_csv_dir: Path) -> float | None:
    """Read the original DPI from a precomputed CSV, if available."""
    csv_path = dpi_csv_dir / (pdf_stem + ".csv")
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        values: list[float] = []
        for col in ("DPI (width)", "DPI (height)"):
            if col in df.columns:
                values.extend(df[col].dropna().tolist())
        if values:
            median = statistics.median(values)
            rounded = int(round(median / 50) * 50)
            return max(rounded, 50)
    except Exception as e:
        print(f"  Warning: failed to read DPI csv {csv_path.name}: {e}")

    return None


def _get_dpi_from_pdf(doc: pymupdf.Document) -> float:
    """Estimate DPI from embedded image metadata in the PDF."""
    dpi_values: list[float] = []
    for page in doc:
        imgs = page.get_images()
        if not imgs:
            continue
        pix = pymupdf.Pixmap(doc, imgs[0][0])
        if pix.xres:
            dpi_values.append(pix.xres)
        elif pix.yres:
            dpi_values.append(pix.yres)

    if dpi_values:
        median = statistics.median(dpi_values)
        rounded = int(round(median / 50) * 50)
        return max(rounded, 50)

    return 150  # fallback


def extract_pdf(
    pdf_path: Path,
    out_dir: Path,
    target_dpi: int = 150,
    dpi_csv_dir: Path | None = None,
) -> str:
    """Extract page images from a single PDF.

    Returns a status string for reporting.
    """
    pdf_stem = pdf_path.stem
    page_dir = out_dir / pdf_stem

    # Check for special-case handler
    if pdf_stem in SPECIAL_CASES:
        handler = SPECIAL_CASES[pdf_stem]
        count = handler(pdf_path, page_dir, target_dpi)
        return f"OK (special): {pdf_stem} — {count} pages"

    # Standard extraction
    doc = pymupdf.open(str(pdf_path))
    page_dir.mkdir(parents=True, exist_ok=True)

    # Determine original DPI
    orig_dpi: float | None = None
    if dpi_csv_dir:
        orig_dpi = _get_dpi_from_csv(pdf_stem, dpi_csv_dir)
    if orig_dpi is None:
        orig_dpi = _get_dpi_from_pdf(doc)

    scale = target_dpi / orig_dpi if orig_dpi else 1.0
    count = 0

    for idx, page in enumerate(doc):
        images = page.get_images()
        if not images:
            continue
        if len(images) != 1:
            print(f"  Warning: {pdf_stem} page {idx} has {len(images)} images (expected 1)")

        pix = pymupdf.Pixmap(doc, images[0][0])

        if abs(scale - 1.0) > 0.01:
            img = _pixmap_to_pil(pix)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), resample=Image.LANCZOS)
            img.save(str(page_dir / f"page_{idx}.png"), dpi=(target_dpi, target_dpi))
        else:
            pix.save(str(page_dir / f"page_{idx}.png"))

        count += 1

    doc.close()
    return f"OK: {pdf_stem} — {count} pages (DPI {orig_dpi}→{target_dpi}, scale {scale:.2f})"


def _extract_wrapper(args):
    """Picklable wrapper for ProcessPoolExecutor."""
    return extract_pdf(*args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert scanned PDFs to normalized PNG page images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs
  python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs --dpi-csv-dir data/corpus-1/dpis
  python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs --workers 8 --skip-existing
        """,
    )
    parser.add_argument(
        "pdf_dir",
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "img_dir",
        help="Destination directory for output image folders",
    )
    parser.add_argument(
        "--dpi-csv-dir",
        help="Directory with precomputed per-PDF DPI CSVs (from compute_dpi.py)",
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
        help="Skip PDFs whose output folder already contains PNGs",
    )
    args = parser.parse_args(argv)

    pdf_dir = Path(args.pdf_dir).resolve()
    img_dir = Path(args.img_dir).resolve()
    dpi_csv_dir = Path(args.dpi_csv_dir).resolve() if args.dpi_csv_dir else None
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    # Validate
    if not pdf_dir.is_dir():
        print(f"ERROR: PDF directory not found: {pdf_dir}", file=sys.stderr)
        return 1

    # Discover PDFs
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {pdf_dir}")
        return 0

    # Filter skip-existing
    if args.skip_existing:
        original = len(pdfs)
        pdfs = [
            p for p in pdfs
            if not (img_dir / p.stem).is_dir()
            or not any((img_dir / p.stem).glob("*.png"))
        ]
        skipped = original - len(pdfs)
        if skipped:
            print(f"Skipping {skipped} PDFs with existing images")

    print(f"PDFs to convert: {len(pdfs)}")
    print(f"Output: {img_dir}")
    print(f"Target DPI: {args.target_dpi}")
    if dpi_csv_dir:
        print(f"DPI CSVs: {dpi_csv_dir}")

    img_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Build tasks
    tasks = [(p, img_dir, args.target_dpi, dpi_csv_dir) for p in pdfs]

    if workers > 1 and len(tasks) > 1:
        print(f"Workers: {workers}")
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_wrapper, t): t[0].name for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(f"FAIL: {futures[future]} — {e}")
    else:
        results = []
        for task in tqdm(tasks, desc="Converting"):
            try:
                results.append(_extract_wrapper(task))
            except Exception as e:
                results.append(f"FAIL: {task[0].name} — {e}")

    # Summary
    ok = sum(1 for r in results if r.startswith("OK"))
    fail = sum(1 for r in results if r.startswith("FAIL"))
    elapsed = time.time() - start

    print(f"\nDone: {ok} succeeded, {fail} failed in {elapsed:.1f}s")
    for r in results:
        if r.startswith("FAIL"):
            print(f"  {r}")

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
