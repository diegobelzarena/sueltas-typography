#!/usr/bin/env python
"""Search for PDF files matching catalogue signatures and optionally copy them.

Given a catalogue CSV (with a ``Signature`` column), this script searches
a PDF directory for matching files and reports results.  With ``--out``,
matched PDFs are copied to a destination folder.

Usage
-----
    # List matches
    python scripts/search_pdfs.py catalogue.csv data/pdfs/

    # Copy matched PDFs to a folder
    python scripts/search_pdfs.py catalogue.csv data/pdfs/ --out data/selected/

    # Quiet mode (summary only)
    python scripts/search_pdfs.py catalogue.csv data/pdfs/ --quiet
"""

import argparse
import csv
import shutil
import sys
import time
from glob import glob
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Signature matching
# ---------------------------------------------------------------------------

def signature_to_pattern(sig: str) -> str:
    """Convert a catalogue signature to a glob pattern.

    Signatures use ``/`` as separator (e.g. ``T/55307/1``) while filenames
    use ``-`` (e.g. ``*T-55307-1.pdf``).
    """
    return f"*{sig.replace('/', '-')}.pdf"


def find_pdfs_for_signatures(
    signatures: list[str],
    pdf_dir: Path,
) -> list[tuple[str, list[Path]]]:
    """Find PDFs in ``pdf_dir`` matching each signature.

    Returns a list of (signature, matched_paths) tuples.
    """
    results = []
    for sig in signatures:
        pattern = signature_to_pattern(sig)
        matches = [Path(p) for p in glob(str(pdf_dir / pattern))]
        results.append((sig, matches))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Search for PDFs matching catalogue signatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/search_pdfs.py catalogue.csv data/pdfs/
  python scripts/search_pdfs.py catalogue.csv data/pdfs/ --out data/selected/
  python scripts/search_pdfs.py catalogue.csv data/pdfs/ --quiet
        """,
    )
    parser.add_argument(
        "catalog",
        help="CSV file with a 'Signature' column",
    )
    parser.add_argument(
        "pdf_dir",
        help="Directory containing PDF files to search",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        help="Copy matched PDFs to this directory",
        default=None,
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print summary, not individual matches",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copying PDFs that already exist in --out directory",
    )
    args = parser.parse_args(argv)

    catalog_path = Path(args.catalog).resolve()
    pdf_dir = Path(args.pdf_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    # Validate
    if not catalog_path.is_file():
        print(f"ERROR: Catalogue not found: {catalog_path}", file=sys.stderr)
        return 1
    if not pdf_dir.is_dir():
        print(f"ERROR: PDF directory not found: {pdf_dir}", file=sys.stderr)
        return 1

    # Read signatures
    signatures = []
    with catalog_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "Signature" not in (reader.fieldnames or []):
            print("ERROR: Catalogue CSV must have a 'Signature' column", file=sys.stderr)
            return 1
        for row in reader:
            sig = row["Signature"].strip()
            if sig:
                signatures.append(sig)

    if not signatures:
        print("No signatures found in catalogue")
        return 0

    print(f"Signatures: {len(signatures)}")
    print(f"PDF dir: {pdf_dir}")

    start = time.time()

    # Search
    results = find_pdfs_for_signatures(signatures, pdf_dir)

    found = 0
    not_found = 0
    copied = 0

    for sig, matches in results:
        if matches:
            found += 1
            if not args.quiet:
                print(f"  {sig}: {len(matches)} file(s)")
                for m in matches:
                    print(f"    {m.name}")

            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                for m in matches:
                    dest = out_dir / m.name
                    if args.skip_existing and dest.exists():
                        continue
                    shutil.copy2(m, dest)
                    copied += 1
        else:
            not_found += 1
            if not args.quiet:
                print(f"  {sig}: NOT FOUND")

    elapsed = time.time() - start

    # Summary
    print(f"\nSummary:")
    print(f"  Found:     {found}/{len(signatures)}")
    print(f"  Not found: {not_found}/{len(signatures)}")
    if out_dir:
        print(f"  Copied:    {copied} files to {out_dir}")
    print(f"  Time:      {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
