"""Helpers for locating PDF files using signatures from a catalog CSV."""

import csv
import os
from glob import glob
from pathlib import Path
from typing import Iterable, List, Tuple


def signature_to_pattern(sig: str) -> str:
    """Convert a signature like ``ABC/123/XYZ`` into a glob pattern.

    The actual files use hyphens in place of slashes, so ``ABC-123-XYZ``
    is searched with a preceding wildcard and a ``.pdf`` suffix.

    Parameters
    ----------
    sig : str
        Signature string from the catalog (three parts separated by
        ``/``).

    Returns
    -------
    pattern : str
        Glob pattern to use when searching for the corresponding PDF.
        Example: ``*ABC-123-XYZ.pdf``
    """
    return f"*{sig.replace('/', '-')}.pdf"


def find_pdfs_for_catalog(
    catalog_csv: Path, pdf_folder: Path
) -> List[Tuple[str, List[Path]]]:
    """Scan ``pdf_folder`` for PDFs matching the signatures listed in
    ``catalog_csv``.

    The CSV must have a header row containing a ``Signature`` column.

    Returns
    -------
    results : list of (signature, matches)
        Each signature is paired with a list of ``Path`` objects for the
        matching PDF files (the list may be empty if no file was found).
    """
    results: List[Tuple[str, List[Path]]] = []

    with catalog_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "Signature" not in reader.fieldnames:
            raise ValueError("catalog CSV does not contain 'Signature' column")
        for row in reader:
            sig = row["Signature"].strip()
            if not sig:
                continue
            pattern = signature_to_pattern(sig)
            matches = [Path(p) for p in glob(str(pdf_folder / pattern))]
            results.append((sig, matches))
    return results


# simple CLI
def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Look up PDFs corresponding to signatures in a CSV file."
    )
    parser.add_argument("catalog", help="Path to the CSV catalog.")
    parser.add_argument("pdf_dir", help="Directory containing PDF files.")
    parser.add_argument(
        "--out",
        dest="out_dir",
        help="Optional directory where matched PDFs will be copied.",
        default=None,
    )
    args = parser.parse_args(argv)

    catalog = Path(args.catalog)
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None

    for sig, matches in find_pdfs_for_catalog(catalog, pdf_dir):
        if matches:
            print(f"{sig}: {len(matches)} file(s)")
            for m in matches:
                print("  ", m)
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    dest = out_dir / m.name
                    # copy file, overwrite if exists
                    import shutil

                    shutil.copy2(m, dest)
        else:
            print(f"{sig}: <no files found>")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
