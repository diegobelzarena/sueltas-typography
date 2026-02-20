#!/usr/bin/env python
"""Command-line wrapper for :mod:`sueltas_typography.pdf_search`."""

import sys
import os

# make sure the package is importable during development
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from sueltas_typography.pdf_search import main


def run():
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    run()
