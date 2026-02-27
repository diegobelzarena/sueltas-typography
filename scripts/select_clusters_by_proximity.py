#!/usr/bin/env python
"""Select italic cluster images that are closest across documents.

This standalone script implements the algorithm described in
`generate_readme_images.py` step 4 but instead of picking the highest-
count cluster per document it chooses, for each letter, the cluster whose
mean image is ``argmin`` to the greatest number of other documents'
clusters.  The ``typographic_distances`` script uses the same
"keep-one-argmin-per-pair" strategy when aggregating distances; here we
re-use exactly that idea to select representative specimens for a grid.

It can optionally produce a grid image similar to the README figure or
simply print/save the chosen cluster indices.

Usage
-----
    python scripts/select_clusters_by_proximity.py CORPUS_DIR [options]

Examples
--------
    python scripts/select_clusters_by_proximity.py data/corpus-2 \
        --letters a,d,e,o,r --plot docs/images/prox_clusters.png

"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os
from typing import Iterable

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

# Printer display: symbol + color, matching the LaTeX convention.
# Keyed by the short printer name (last token of the full name).
STEP4_PRINTER_STYLE: dict[str, tuple[str, str]] = {
    "Ábrego":  ("\u2020", "tab:blue"),    # † blue  — Rodríguez de Ábrego
    "Lyra":    ("\u2021", "tab:orange"),   # ‡ orange — Lyra
}

# defer heavy imports until needed for plotting

def _load_doc_index_map(csv_path: Path | None) -> dict[str, int]:
    """Return mapping {folder_name: index} from a metadata CSV.

    The CSV is expected to have a column named "FileName" and a column
    named "Index" (as used throughout the project).  If the file is
    missing or malformed we return an empty dict.
    """
    mapping: dict[str, int] = {}
    if csv_path is None or not csv_path.exists():
        return mapping
    try:
        import csv as _csv
    except ImportError:
        return mapping
    with open(csv_path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            fname = row.get("FileName", "")
            try:
                mapping[fname] = int(row.get("Index", "")) - 1
            except ValueError:
                continue
    return mapping


def _load_doc_printer_map(csv_path: Path | None) -> dict[str, str]:
    """Return {FileName: Printer} from the ordered-table CSV."""
    mapping: dict[str, str] = {}
    if csv_path is None or not csv_path.exists():
        return mapping
    try:
        import csv as _csv
    except ImportError:
        return mapping
    with open(csv_path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            fname = row.get("FileName", "")
            printer = row.get("Printer", "").strip()
            if fname:
                mapping[fname] = printer if printer else "unknown"
    return mapping

def find_documents(corpus_dir: Path, index_map: dict[str, int] | None = None) -> list[Path]:
    """Return list of subdirectories containing cluster data.

    If *index_map* is given the returned list is sorted by the map (then
    by name) mimicking the CSV ordering used elsewhere in the project.
    """
    dirs = [d for d in corpus_dir.iterdir() if d.is_dir() and (d / "clusters_all.npz").exists()]
    if index_map:
        dirs = sorted(dirs, key=lambda d: (index_map.get(d.name, 9999), d.name))
    else:
        dirs = sorted(dirs)
    return dirs


def _collect_candidates(
    doc_dirs: list[Path], letter: str, italic_thresh: float = 0.5
):
    """Gather all italic clusters for *letter*.

    Returns
    -------
    means : np.ndarray, shape (N, H*W)
        Flattened cluster-mean images.
    doc_ids : np.ndarray, shape (N,)
        Document index for each candidate.
    cluster_ids : np.ndarray, shape (N,)
        Original cluster index within the document.
    shapes : list[tuple]
        Shape of each mean image (for un-flattening).
    """
    means = []
    doc_ids = []
    cluster_ids = []
    shapes: list[tuple] = []

    for did, d in enumerate(doc_dirs):
        cp = d / "clusters_all.npz"
        if not cp.exists():
            continue
        data = np.load(str(cp), allow_pickle=True)
        lbls = data["cluster_labels"][:, 0]
        italics = data["cluster_italic"]
        mask = (lbls == letter) & (italics > italic_thresh)
        for cid in np.where(mask)[0]:
            m = data["cluster_means"][cid]
            means.append(m.ravel())
            doc_ids.append(did)
            cluster_ids.append(cid)
            shapes.append(m.shape)

    if not means:
        return np.empty((0, 0)), np.empty((0,), int), np.empty((0,), int), []

    return np.vstack(means), np.array(doc_ids, dtype=int), np.array(cluster_ids), shapes


def _choose_best_per_doc(
    means: np.ndarray,
    doc_ids: np.ndarray,
    strategy: str = "argmin",
) -> dict[int, int]:
    """Return a map from document index to winning candidate index.

    Two strategies are supported:

    * ``argmin`` (default) – replicate the previous behaviour: tally
      wins for every pairwise comparison between different documents and
      keep the candidate with the most wins per document.
    * ``median`` – for each candidate compute the median distance to all
      candidates in other documents, then choose the candidate with the
      smallest median within each document.
    """
    n = len(means)
    if n == 0:
        return {}

    dists = pairwise_distances(means, metric="cosine")

    if strategy == "argmin":
        counts = np.zeros(n, dtype=int)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if doc_ids[i] == doc_ids[j]:
                    continue
                winner = i if dists[i, j] <= dists[j, i] else j
                counts[winner] += 1
        best: dict[int, int] = {}
        for idx, doc in enumerate(doc_ids):
            if doc not in best or counts[idx] > counts[best[doc]]:
                best[doc] = idx
        return best

    elif strategy == "median":
        # compute median distance to all candidates in other docs for each
        # candidate
        medians = np.full(n, np.inf)
        for i in range(n):
            mask = doc_ids != doc_ids[i]
            if np.any(mask):
                medians[i] = np.quantile(dists[i, mask], 0.25)
        best: dict[int, int] = {}
        for idx, doc in enumerate(doc_ids):
            if doc not in best or medians[idx] < medians[best[doc]]:
                best[doc] = idx
        return best

    else:
        raise ValueError(f"unknown strategy '{strategy}'")


def select_clusters(
    corpus_dir: Path,
    letters: Iterable[str],
    italic_thresh: float = 0.5,
    index_map: dict[str, int] | None = None,
    strategy: str = "argmin",
):
    """Compute representative images for each letter/document.

    The documents are listed from *corpus_dir* and optionally ordered
    according to *index_map* (returned by ``_load_doc_index_map``).

    Returns ``(doc_dirs, grid)`` where ``grid[letter][doc]`` is either an
    array or ``None``.
    """
    doc_dirs = find_documents(corpus_dir, index_map)
    grid: dict[str, list[np.ndarray | None]] = {}

    for letter in letters:
        means, doc_ids, cluster_ids, shapes = _collect_candidates(
            doc_dirs, letter, italic_thresh
        )
        best_map = _choose_best_per_doc(means, doc_ids, strategy=strategy)
        row: list[np.ndarray | None] = [None] * len(doc_dirs)
        for doc, cand_idx in best_map.items():
            row[doc] = means[cand_idx].reshape(shapes[cand_idx])
        grid[letter] = row

    return doc_dirs, grid


def _plot_grid(
    doc_dirs: list[Path],
    grid: dict[str, list[np.ndarray | None]],
    letters: list[str],
    out_path: Path | None = None,
    doc_labels: list[str] | None = None,
    doc_symbols: list[tuple[str, str]] | None = None,
    save_fig: bool = True,
) -> plt.Figure:
    """Create (and optionally save) a simple grid of selected images.

    *doc_labels* provides a string to show above each document column.
    *doc_symbols* is a list of (symbol, color) tuples matching the docs.

    Parameters
    ----------
    out_path
        Base path where PNG/SVG will be written.  Only used if *save_fig*
        is ``True``; if ``None`` the figure is not written even when
        *save_fig* is ``True``.
    save_fig
        If ``False``, the figure will be returned but not saved; the caller
        can modify it (add suptitle, annotations) and save manually.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nletters = len(letters)
    ndocs = len(doc_dirs)

    cell_px = 0.55
    header_row_h = 0.35

    fig_w = ndocs * cell_px + 0.3
    fig_h = header_row_h + nletters * cell_px + 0.2

    fig, axes = plt.subplots(
        nletters, ndocs,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "width_ratios": [1] * ndocs,
            "wspace": -0.20,
            # reduce vertical padding between rows
            "hspace": -0.25,
        },
    )
    # plt.subplots can return a 1-d array when cols == 1; ensure a 2-d
    # matrix shaped [nletters, ndocs+1].  We want rows==nletters.
    axes = np.array(axes)
    if axes.ndim == 1:
        # single column case -> shape (nletters,), convert to (nletters,1)
        axes = axes[:, None]
    elif axes.ndim == 2 and axes.shape[0] != nletters:
        # if dims swapped (happens when nletters==1), transpose
        axes = axes.T

    for r, letter in enumerate(letters):
        for c in range(ndocs):
            ax = axes[r, c]
            img = grid[letter][c]
            if img is not None:
                ax.imshow(img, cmap="gray_r")
            else:
                # draw a shorter gray square centred in the cell
                import matplotlib.patches as _patches
                rect = _patches.Rectangle(
                    (0.25, 0.1), 0.5, 0.65,
                    transform=ax.transAxes,
                    facecolor="#f5f5f5"
                )
                ax.add_patch(rect)
                ax.set_facecolor("white")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            # per-column header (first row only)
            if r == 0 and doc_labels is not None:
                num = doc_labels[c]
                sym_char, sym_color = (doc_symbols[c] if doc_symbols
                                       else ("", "black"))
                # draw the index number in black with a colored circle around it
                bbox = None
                if sym_char:
                    bbox = dict(boxstyle=f"circle,pad={0.2 if int(num) >= 10 else 0.4}",
                                edgecolor=sym_color, facecolor="white",
                                linewidth=1.2)
                ax.text(
                    0.5, 1.14, num,
                    transform=ax.transAxes,
                    fontsize=10, fontfamily="serif",
                    ha="center", va="bottom", color="black",
                    bbox=bbox,
                )

    # fig.tight_layout()
    if save_fig and out_path is not None:
        # save in both png and svg formats
        base = str(out_path)
        root, ext = os.path.splitext(base)
        for fmt in ("png", "svg"):
            fname = root + "." + fmt
            fig.savefig(fname, bbox_inches="tight")
            print(f"Saved grid to {fname}")
    return fig



def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Pick italic clusters closest to others across documents."
    )
    parser.add_argument("corpus", type=Path, help="path to corpus directory")
    parser.add_argument(
        "--letters", type=str, help="comma-separated letters to process",
    )
    parser.add_argument(
        "--plot", type=Path, help="output path for grid image (optional)"
    )
    parser.add_argument(
        "--csv", type=Path,
        help="metadata CSV used to order documents (FileName/Index columns)"
    )
    parser.add_argument(
        "--strategy", choices=["argmin", "median"], default="argmin",
        help="cluster selection strategy: argmin (default) or median"
    )
    parser.add_argument(
        "--italic-threshold", type=float, default=0.5,
        help="minimum italic score to include a cluster",
    )
    args = parser.parse_args(argv)

    letters: list[str]
    if args.letters:
        letters = [l for l in args.letters.split(",") if l]
    else:
        # default example set similar to generate_readme_images
        letters = ["a", "d", "e", "o", "r"]

    index_map = _load_doc_index_map(args.csv) if args.csv else None
    printer_map = _load_doc_printer_map(args.csv) if args.csv else {}

    doc_dirs, grid = select_clusters(
        args.corpus, letters, italic_thresh=args.italic_threshold,
        index_map=index_map,
        strategy=args.strategy,
    )

    # build display labels/symbols based on metadata
    doc_labels: list[str] = []
    doc_symbols: list[tuple[str, str]] = []
    for d in doc_dirs:
        idx = index_map.get(d.name) if index_map else None
        doc_labels.append(str(idx) if idx is not None else d.name[:8])
        full_printer = printer_map.get(d.name, "unknown")
        short = full_printer.split()[-1] if full_printer else "unknown"
        style = STEP4_PRINTER_STYLE.get(short)
        if style is not None:
            doc_symbols.append(style)
        elif full_printer != "unknown" and full_printer:
            doc_symbols.append(("*", "black"))
        else:
            doc_symbols.append(("*", "#999"))

    # print summary table
    print("document\tletter\tchosen_cluster")
    for did, d in enumerate(doc_dirs):
        for letter in letters:
            row = grid[letter][did]
            chosen = "none" if row is None else "present"
            print(f"{d.name}\t{letter}\t{chosen}")

    if args.plot:
        if doc_dirs:
            _plot_grid(
                doc_dirs, grid, letters, args.plot,
                doc_labels=doc_labels, doc_symbols=doc_symbols,
            )
        else:
            print("no documents found; skipping plot")


if __name__ == "__main__":
    sys.exit(main())
