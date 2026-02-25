#!/usr/bin/env python
"""Compute DPI and physical dimensions for PDF and TIFF image collections.

Processes a directory of PDF files or TIFF-image folders and produces
CSV reports with DPI, page dimensions (inches/cm), and source metadata.

Usage
-----
    # Single CSV output for all PDFs
    python scripts/compute_dpi.py data/corpus-1/pdfs -o dpi_report.csv

    # One CSV per source (for use with convert_sources.py --dpi-csv-dir)
    python scripts/compute_dpi.py data/corpus-1/pdfs -o data/corpus-1/dpis --per-file

    # Process TIFF folders
    python scripts/compute_dpi.py data/corpus-1/tiffs -o dpi_report.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# DPI extraction
# ---------------------------------------------------------------------------

COLUMN_ORDER = [
    "suelta",
    "page",
    "DPI (width)",
    "DPI (height)",
    "width (in)",
    "height (in)",
    "width (cm)",
    "height (cm)",
    "Source of physical dimension",
]


def get_tiff_info(file_path: Path) -> dict:
    """Extract DPI and dimensions from a TIFF file."""
    with Image.open(file_path) as img:
        w_px, h_px = img.size
        dpi = img.info.get("dpi", (None, None))
        dpi_w, dpi_h = dpi

        w_in = w_px / dpi_w if dpi_w else None
        h_in = h_px / dpi_h if dpi_h else None

        return {
            "DPI (width)": float(dpi_w) if dpi_w else None,
            "DPI (height)": float(dpi_h) if dpi_h else None,
            "width (in)": w_in,
            "height (in)": h_in,
            "width (cm)": w_in * 2.54 if w_in else None,
            "height (cm)": h_in * 2.54 if h_in else None,
            "Source of physical dimension": "DPI (TIF metadata)",
        }


def get_pdf_info(pdf_path: Path) -> list[dict]:
    """Extract page dimensions and effective DPI from a PDF.

    Returns one dict per page with DPI estimated from the ratio of
    embedded image pixels to the PDF page size (in points → inches).
    """
    pages_data = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            w_in = page.rect.width / 72
            h_in = page.rect.height / 72

            images = page.get_images()
            if not images:
                continue

            xref = images[0][0]
            pix = pymupdf.Pixmap(doc, xref)
            w_px, h_px = pix.width, pix.height

            dpi_w = w_px / w_in if w_in else None
            dpi_h = h_px / h_in if h_in else None

            pages_data.append({
                "page": f"page_{i}.png",
                "DPI (width)": dpi_w,
                "DPI (height)": dpi_h,
                "width (in)": w_in,
                "height (in)": h_in,
                "width (cm)": w_in * 2.54,
                "height (cm)": h_in * 2.54,
                "Source of physical dimension": "size in inches (PDF metadata)",
            })
    return pages_data


def process_item(item: Path) -> list[dict]:
    """Process a single PDF or TIFF folder, returning rows of metadata."""
    rows = []
    suelta_name = item.name

    if item.is_file() and item.suffix.lower() == ".pdf":
        try:
            for row in get_pdf_info(item):
                row["suelta"] = suelta_name
                rows.append(row)
        except Exception as e:
            print(f"  Error processing PDF {item.name}: {e}")

    elif item.is_dir():
        tiffs = sorted(
            f for f in item.iterdir()
            if f.suffix.lower() in (".tif", ".tiff")
        )
        for tiff_file in tiffs:
            try:
                row = get_tiff_info(tiff_file)
                row["suelta"] = suelta_name
                row["page"] = tiff_file.name
                rows.append(row)
            except Exception as e:
                print(f"  Error processing TIFF {tiff_file.name}: {e}")

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute DPI and physical dimensions for PDF/TIFF collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compute_dpi.py data/corpus-1/pdfs -o dpi_report.csv
  python scripts/compute_dpi.py data/corpus-1/pdfs -o data/corpus-1/dpis --per-file
  python scripts/compute_dpi.py data/corpus-1/tiffs -o dpi_report.csv
        """,
    )
    parser.add_argument(
        "root",
        help="Directory containing PDFs or TIFF folders",
    )
    parser.add_argument(
        "--output", "-o",
        default="dpi.csv",
        help="Output CSV file or directory (with --per-file). Default: dpi.csv",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Write a separate CSV for each PDF/TIFF folder",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip items that already have a CSV in the output directory (--per-file only)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    out = Path(args.output)

    if not root.is_dir():
        print(f"ERROR: Directory not found: {root}", file=sys.stderr)
        return 1

    # Discover items
    items = sorted(
        item for item in root.iterdir()
        if item.is_dir() or (item.is_file() and item.suffix.lower() == ".pdf")
    )
    if not items:
        print(f"No PDFs or folders found in {root}")
        return 0

    start = time.time()
    print(f"Items to process: {len(items)}")

    if args.per_file:
        out.mkdir(parents=True, exist_ok=True)
        written = 0

        for item in tqdm(items, desc="Computing DPI"):
            stem = item.stem if item.is_file() else item.name
            csv_name = stem + ".csv"

            if args.skip_existing and (out / csv_name).exists():
                continue

            rows = process_item(item)
            if rows:
                df = pd.DataFrame(rows).reindex(columns=COLUMN_ORDER)
                df.to_csv(out / csv_name, index=False)
                written += 1

        elapsed = time.time() - start
        print(f"\nDone: wrote {written} CSV files to {out} in {elapsed:.1f}s")
    else:
        all_rows = []
        for item in tqdm(items, desc="Computing DPI"):
            all_rows.extend(process_item(item))

        df = pd.DataFrame(all_rows).reindex(columns=COLUMN_ORDER)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        elapsed = time.time() - start
        print(f"\nDone: wrote {len(df)} rows to {out} in {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
