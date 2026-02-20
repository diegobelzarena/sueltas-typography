#!/usr/bin/env python
"""Thin wrapper script that exposes :mod:`sueltas_typography.pdf_utils`.

Usage:

    python scripts/convert_pdfs.py /path/to/pdfs /path/to/images

"""

import sys
import os

# ensure the package can be imported when running the script directly
# (e.g. during development).  The project uses a `src/` layout so we
# add it to the path if it's not already there.
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from sueltas_typography.pdf_utils import main


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
