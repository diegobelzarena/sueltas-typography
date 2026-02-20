"""Utilities for converting PDF scans to PNG page images."""

import argparse
import os
from os.path import join
from typing import Iterable

import pymupdf


def extract_images(pdf_path: str, out_dir: str, report_page_info: bool = False) -> None:
    """Extract the (single) scan image from each page of ``pdf_path``.

    The output images are saved as ``page_{idx}.png`` inside ``out_dir``.
    ``out_dir`` is created if it does not exist.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file containing scanned pages.
    out_dir : str
        Directory where PNGs will be written.  Existing files may be
        overwritten.
    report_page_info : bool
        If ``True`` print DPI and pixel dimensions for each page.
    """
    doc = pymupdf.open(pdf_path)
    os.makedirs(out_dir, exist_ok=True)

    for idx, page in enumerate(doc):
        images = page.get_images()
        if len(images) != 1:
            print(
                f"Warning: file {os.path.basename(pdf_path)}, page {idx} "
                f"has {len(images)} images (expected 1)."
            )
        xref = images[0][0]
        pix = pymupdf.Pixmap(doc, xref)
        if report_page_info:
            # print whatever metadata is attached to the page; size or other
            # attributes may already be present in that dictionary, so we
            # defer to whatever the PDF provides rather than fabricating
            # values ourselves.
            try:
                pmeta = page.metadata
            except AttributeError:
                pmeta = {}
            print(f"page {idx} metadata: {pmeta}")
        pix.save(join(out_dir, f"page_{idx}.png"))


def convert_directory(pdf_dir: str, img_dir: str, check_metadata: bool = False, report_page_info: bool = False) -> None:
    """Walk ``pdf_dir`` and convert every ``.pdf`` file to a subfolder of
    ``img_dir``.

    Each PDF ``foo.pdf`` yields a subdirectory ``img_dir/foo/`` containing
    the extracted pages.  If ``check_metadata`` is ``True`` the metadata
    is printed before conversion.
    """
    os.makedirs(img_dir, exist_ok=True)
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = join(pdf_dir, filename)
            if check_metadata:
                meta = get_pdf_metadata(pdf_path)
                print(f"{filename} metadata: {meta}")
            out_sub = join(img_dir, filename[:-4])
            extract_images(pdf_path, out_sub, report_page_info=report_page_info)
        else:
            print(f"Skipping non-PDF file {filename}")


def get_pdf_metadata(pdf_path: str) -> dict:
    """Return the metadata dictionary from a PDF file.

    Uses :mod:`pymupdf` to open the document and read its ``metadata``
    attribute.  This can include title, author, creation/modification
    dates, etc., depending on the source of the PDF.
    """
    doc = pymupdf.open(pdf_path)
    return doc.metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a directory of scanned PDFs to PNG page images."
    )
    parser.add_argument("pdf_dir", help="Directory containing PDF files.")
    parser.add_argument(
        "img_dir", help="Destination directory for output image folders."
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Print PDF metadata before converting each file.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print DPI/size info for each page.",
    )
    args = parser.parse_args(argv)
    convert_directory(
        args.pdf_dir,
        args.img_dir,
        check_metadata=args.metadata,
        report_page_info=args.info,
    )


if __name__ == "__main__":
    main()
