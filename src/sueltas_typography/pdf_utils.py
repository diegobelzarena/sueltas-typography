"""Utilities for converting PDF scans to PNG page images."""

import argparse
import os
from os.path import join
from typing import Iterable, Optional
import statistics

import pandas as pd
import numpy as np
from PIL import Image
import pymupdf


def _pixmap_to_pil(pix: pymupdf.Pixmap) -> Image.Image:
    """Convert a PyMuPDF Pixmap to a Pillow Image."""
    if pix.n >= 3:
        mode = "RGB"
    else:
        mode = "L"
    if pix.alpha:
        mode += "A"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def extract_images(
    pdf_path: str, out_dir: str, dpi_csv_dir: Optional[str] = None
) -> None:
    """Extract and normalise page images from ``pdf_path``.

    A DPI value is determined either by reading a CSV file named after the
    PDF in ``dpi_csv_dir`` (if provided) or by looking at the embedded
    image resolutions.  The resulting DPI is rounded to the nearest
    multiple of 50 and used to compute a scaling factor that brings all
    images to 150 dpi.
    """
    # special-case handling for BNE_796_587_T-55352-9
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    if basename == "BNE_796_587_T-55352-9":
        _extract_special(pdf_path, out_dir)
        return

    doc = pymupdf.open(pdf_path)
    os.makedirs(out_dir, exist_ok=True)

    orig_dpi: Optional[float] = None
    if dpi_csv_dir:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        csv_path = join(dpi_csv_dir, stem + ".csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                values: list[float] = []
                for col in ("DPI (width)", "DPI (height)"):
                    if col in df.columns:
                        values.extend(df[col].dropna().tolist())
                if values:
                    median = statistics.median(values)
                    orig_dpi = int(round(median / 50) * 50)
                    if orig_dpi < 50:
                        orig_dpi = 50
            except Exception as e:
                print(f"warning: failed to read DPI csv {csv_path}: {e}")
    if orig_dpi is None:
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
            orig_dpi = int(round(median / 50) * 50)
            if orig_dpi < 50:
                orig_dpi = 50
        else:
            orig_dpi = 150

    scale = 150 / orig_dpi if orig_dpi and orig_dpi != 0 else 1.0

    for idx, page in enumerate(doc):
        images = page.get_images()
        if len(images) != 1:
            print(
                f"Warning: file {os.path.basename(pdf_path)}, page {idx} "
                f"has {len(images)} images (expected 1)."
            )
        if not images:
            continue
        pix = pymupdf.Pixmap(doc, images[0][0])
        if scale != 1.0:
            img = _pixmap_to_pil(pix)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), resample=Image.LANCZOS)
            img.save(join(out_dir, f"page_{idx}.png"), dpi=(150, 150))
        else:
            pix.save(join(out_dir, f"page_{idx}.png"))


def _extract_special(pdf_path: str, out_dir: str) -> None:
    """Custom extraction logic for BNE_796_587_T-55352-9.

    This replicates the behaviour from the standalone ``create_imgs_lyra3``
    script: combine background and text layers with a mask.
    """
    doc = pymupdf.open(pdf_path)
    os.makedirs(out_dir, exist_ok=True)
    for idx, page in enumerate(doc):
        images = page.get_images()
        if not images:
            continue
        # background image is first
        image = images[0]
        pix = pymupdf.Pixmap(doc, image[0])
        h, w, c = pix.height, pix.width, pix.n
        img0 = np.frombuffer(pix.samples, dtype=np.uint8).reshape((h, w, c))
        img = img0.copy()
        # text image is second with mask
        if len(images) > 1:
            image = images[1]
            xref = image[0]
            smask = image[1]
            pix2 = pymupdf.Pixmap(doc, xref)
            h1, w1, _ = pix2.height, pix2.width, pix2.n
            img_pil = Image.frombytes(mode='RGB', size=(w1, h1), data=pix2.samples)
            img1 = np.array(img_pil.resize((w, h)))
            pixm = pymupdf.Pixmap(doc, smask)
            if (h, w) != (pixm.height, pixm.width):
                print(f"Warning: file {os.path.basename(pdf_path)}, page {idx} "
                      "has different sizes for background image and text mask.")
            else:
                mask = np.frombuffer(pixm.samples, dtype=np.bool_).reshape((h, w))
                img[mask] = img1[mask]
        # save using PIL instead of imageio
        pil_out = Image.fromarray(img)
        pil_out.save(join(out_dir, f'page_{idx}.png'))


def convert_directory(pdf_dir: str, img_dir: str, dpi_csv_dir: Optional[str] = None) -> None:
    """Convert all PDFs in ``pdf_dir`` to PNG folders in ``img_dir``."""
    os.makedirs(img_dir, exist_ok=True)
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = join(pdf_dir, filename)
            out_sub = join(img_dir, filename[:-4])
            # skip if already done
            if os.path.isdir(out_sub) and any(f.lower().endswith('.png') for f in os.listdir(out_sub)):
                print(f"Skipping {filename}: images already exist")
                continue
            extract_images(pdf_path, out_sub, dpi_csv_dir=dpi_csv_dir)
        else:
            print(f"Skipping non-PDF file {filename}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a directory of scanned PDFs to PNG page images."
    )
    parser.add_argument("pdf_dir", help="Directory containing PDF files.")
    parser.add_argument(
        "img_dir", help="Destination directory for output image folders."
    )
    parser.add_argument(
        "--dpi-csv-dir",
        help="Directory containing per-pdf CSVs with precomputed DPI values.",
        default=None,
    )
    args = parser.parse_args(argv)
    convert_directory(
        args.pdf_dir,
        args.img_dir,
        dpi_csv_dir=args.dpi_csv_dir,
    )


if __name__ == "__main__":
    main()
