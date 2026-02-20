"""Utilities for extracting DPI and size information from PDFs and TIFFs.

The original script provided by the user has been refactored into this
module so it can be imported from other parts of the project.
"""

import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from PIL import Image
import pymupdf


def get_tiff_info(file_path: Path) -> Dict[str, object]:
    """Extracts DPI and dimensions from a TIFF file metadata."""
    with Image.open(file_path) as img:
        w_px, h_px = img.size
        # DPI metadata may be missing or stored in different keys; PIL puts
        # it in ``img.info['dpi']`` when available.
        dpi = img.info.get("dpi", (None, None))
        dpi_w, dpi_h = dpi

        # width/height in inches based on DPI if available
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


def get_pdf_info(pdf_path: Path) -> List[Dict[str, object]]:
    """Extracts page dimensions from a PDF and calculates effective DPI.

    The function returns a list of dictionaries, one per page.  The
    rectangular size of the page comes from ``page.rect`` (in points);
    the pixel dimensions of the first image on the page (if any) are used
    to estimate the DPI.
    """
    pages_data: List[Dict[str, object]] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            w_in = page.rect.width / 72
            h_in = page.rect.height / 72

            images = page.get_images()
            if images:
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


def _rows_for_item(item: Path) -> List[Dict[str, object]]:
    """Return a list of metadata rows for a single item (PDF or TIFF folder)."""
    rows: List[Dict[str, object]] = []
    suelta_name = item.name
    if item.is_file() and item.suffix.lower() == ".pdf":
        try:
            pdf_results = get_pdf_info(item)
            for page_row in pdf_results:
                page_row.update({"suelta": suelta_name})
                rows.append(page_row)
        except Exception as e:
            print(f"Error processing PDF {item}: {e}")
    elif item.is_dir():
        tiffs = sorted(
            [f for f in item.iterdir() if f.suffix.lower() in [".tif", ".tiff"]]
        )
        for tiff_file in tiffs:
            try:
                tiff_data = get_tiff_info(tiff_file)
                tiff_data.update({
                    "suelta": suelta_name,
                    "page": tiff_file.name,
                })
                rows.append(tiff_data)
            except Exception as e:
                print(f"Error processing TIFF {tiff_file}: {e}")
    return rows


def process_all_files(root_path: Path) -> pd.DataFrame:
    """Walk a single directory of PDFs or TIFF folders and gather DPI info.

    The directory should contain either PDF files or subdirectories with
    TIFF images; there is no additional batch layer.

    Returns a DataFrame with one row per page or TIFF and columns suitable
    for export to CSV.
    """
    all_rows: List[Dict[str, object]] = []
    root = Path(root_path)

    for item in sorted(root.iterdir()):
        if not item.exists():
            continue
        all_rows.extend(_rows_for_item(item))

    # build a dataframe; if ``all_rows`` is empty the result has no
    # columns, so we reindex afterwards to ensure the expected columns
    # exist (populated with NaN).
    df = pd.DataFrame(all_rows)
    column_order = [
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
    return df.reindex(columns=column_order)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk batches of PDFs/TIFFs and produce a CSV of DPI/size info."
    )
    parser.add_argument("root", help="Directory containing PDFs or TIFF folders.")
    parser.add_argument(
        "--output", "-o", default="dpi.csv", help="Output CSV filename or directory."
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Write a separate CSV for each PDF/tiff folder instead of one big file.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.output)

    if args.per_file:
        # ensure output is a directory
        out.mkdir(parents=True, exist_ok=True)
        for item in sorted(root.iterdir()):
            rows = _rows_for_item(item)
            df = pd.DataFrame(rows)
            df = df.reindex(columns=[
                "suelta",
                "page",
                "DPI (width)",
                "DPI (height)",
                "width (in)",
                "height (in)",
                "width (cm)",
                "height (cm)",
                "Source of physical dimension",
            ])
            if not df.empty:
                # remove suffix if present so pdf -> foo.csv instead of foo.pdf.csv
                stem = item.stem if item.is_file() else item.name
                csv_name = stem + ".csv"
                df.to_csv(out / csv_name, index=False)
                print(f"wrote {len(df)} rows to {out / csv_name}")
        return 0
    else:
        df = process_all_files(root)
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows to {out}")
        return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
