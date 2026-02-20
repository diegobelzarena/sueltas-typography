#!/usr/bin/env python
"""Script wrapping :mod:`sueltas_typography.dpi_info` to create a CSV."""

import sys
import os

# ensure src path for imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from sueltas_typography.dpi_info import main


def run():
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    run()
