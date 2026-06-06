#!/usr/bin/env python
"""Export a contrario results and glyph images for the sueltas-app dashboard.

Reads the outputs of the typography pipeline (``acontrario_results.npz``,
``distances_*.npz``, per-document ``clusters_all.npz``) and produces all the
data files that sueltas-app expects:

  • ``n1hat_combined_matrix_ordered.npy``  — Combined n̂₁ matrix (OLO-reordered)
  • ``n1hat_rm_matrix_ordered.npy``  — Roman n̂₁ matrix (OLO-reordered)
  • ``n1hat_it_matrix_ordered.npy``  — Italic n̂₁ matrix (OLO-reordered)
  • ``w_combined_matrix_ordered.npy``  — Combined weight matrix (OLO-reordered)
  • ``w_rm_matrix_ordered.npy``      — Roman weight matrix (OLO-reordered)
  • ``w_it_matrix_ordered.npy``      — Italic weight matrix (OLO-reordered)
  • ``books_dashboard_ordered.npy``  — Book IDs in display order
  • ``impr_names_dashboard_ordered.npy`` — Printer names in display order
  • ``symbs_dashboard.npy``          — Unique letter symbols analysed
  • ``images/<book_id>/*.webp``      — Glyph images per book
  • ``images/images_cache_meta.pkl`` — Image metadata index

Usage
-----
    python scripts/export_to_dashboard.py data/corpus-2 \\
        --config configs/acontrario_corpus2.yaml \\
        --out /path/to/sueltas-app/data

    # Skip image export (matrices only)
    python scripts/export_to_dashboard.py data/corpus-2 \\
        --config configs/acontrario_corpus2.yaml \\
        --out /path/to/sueltas-app/data --skip-images
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from acontrario import hierarchical_olo_order

# ---------------------------------------------------------------------------
# Default a contrario config (only the keys we need)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "analysis": {
        "metric": "average",
        "ordering": "hierarchical",
    },
}

# Thresholds matching typographic_distances.yaml defaults
ROMAN_ITALIC_MAX = 0.1
ITALIC_ITALIC_MIN = 0.4
LABEL_CONFIDENCE_MIN = 0.6
TOP_CLUSTERS_PER_LETTER = 5

def _is_letter(ch: str) -> bool:
    """Return True for letters (including accented), False for punctuation/digits."""
    return len(ch) == 1 and ch.isalpha()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str | None) -> dict:
    """Load and merge a YAML config with defaults."""
    cfg = DEFAULT_CONFIG.copy()
    if config_path and os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # Merge analysis section
        if "analysis" in user_cfg:
            cfg["analysis"] = {**cfg["analysis"], **user_cfg["analysis"]}
    return cfg


def _compute_ordering(
    n1hat_rm: np.ndarray | None,
    n_rm: np.ndarray | None,
    n1hat_it: np.ndarray | None,
    n_it: np.ndarray | None,
    metric_mode: str,
    ordering_mode: str,
) -> np.ndarray:
    """Compute the OLO or index-based ordering.

    Replicates the logic from run_acontrario.py lines 406–466.
    Returns an index array ``order`` such that ``books[order]`` gives
    the display order.
    """
    # Weight matrices: w = n1hat / n where n1hat > 0, else 0
    w_rm = (np.where(n1hat_rm > 0, n1hat_rm / n_rm, 0)
            if n1hat_rm is not None else None)
    w_it = (np.where(n1hat_it > 0, n1hat_it / n_it, 0)
            if n1hat_it is not None else None)

    # Select which weight drives ordering
    if metric_mode == "roman":
        w_order = w_rm if w_rm is not None else w_it
    elif metric_mode == "italic":
        w_order = w_it if w_it is not None else w_rm
    else:  # "average"
        if w_rm is not None and w_it is not None:
            w_order = (w_rm + w_it) / 2
        elif w_rm is not None:
            w_order = w_rm
        else:
            w_order = w_it

    metric = np.clip(1 - w_order, 0, 1)

    if ordering_mode == "hierarchical":
        order = hierarchical_olo_order(metric)
    else:
        # Identity order (documents stay in their original order)
        order = np.arange(len(metric))

    return order, w_rm, w_it


def _reorder_matrix(mat: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Reorder a square matrix by ``order`` on both axes."""
    mesh = np.meshgrid(order, order)
    return mat[mesh[0], mesh[1]]


# ---------------------------------------------------------------------------
# Glyph image export
# ---------------------------------------------------------------------------

def _export_glyph_images(
    charnet_dir: Path,
    books: np.ndarray,
    out_images_dir: Path,
    top_n: int = TOP_CLUSTERS_PER_LETTER,
) -> dict:
    """Export cluster mean images as WebP files.

    Returns the ``book_index`` dict for ``images_cache_meta.pkl``.
    """
    from PIL import Image

    book_index: dict[str, list[tuple[str, str]]] = {}
    all_letters_set: set[str] = set()

    for book_name in books:
        book_name = str(book_name)  # ensure plain Python str (not np.str_)
        cluster_path = charnet_dir / book_name / "clusters_all.npz"
        if not cluster_path.exists():
            print(f"  Warning: No clusters for {book_name}, skipping images")
            continue

        data = np.load(str(cluster_path), allow_pickle=True)
        cluster_means = data["cluster_means"]
        cluster_labels = data["cluster_labels"]
        cluster_italic = data["cluster_italic"]

        if len(cluster_means) == 0:
            continue

        book_out = out_images_dir / book_name
        book_out.mkdir(parents=True, exist_ok=True)
        book_entries: list[tuple[str, str]] = []

        # Process both styles
        for font_type, italic_filter in [("roman", "max"), ("italic", "min")]:
            if italic_filter == "max":
                style_mask = cluster_italic <= ROMAN_ITALIC_MAX
            else:
                style_mask = cluster_italic >= ITALIC_ITALIC_MIN

            # Label confidence filter
            conf_mask = np.array([
                float(cl[1]) >= LABEL_CONFIDENCE_MIN for cl in cluster_labels
            ])

            combined_mask = style_mask & conf_mask
            if not combined_mask.any():
                continue

            selected_indices = np.where(combined_mask)[0]
            selected_labels = np.array([str(cluster_labels[i][0]) for i in selected_indices])

            # Keep top N per letter by sample count
            unique_letters = np.unique(selected_labels)
            for letter in unique_letters:
                # Only export actual letters (skip punctuation, digits, etc.)
                if not _is_letter(letter):
                    continue

                letter_mask = selected_labels == letter
                letter_indices = selected_indices[letter_mask]

                if len(letter_indices) > top_n:
                    counts = [int(cluster_labels[i][2]) for i in letter_indices]
                    top_idx = np.argsort(counts)[-top_n:]
                    letter_indices = letter_indices[top_idx]

                # Determine case prefix
                if letter.isupper():
                    case_prefix = "upper"
                else:
                    case_prefix = "lower"

                for idx_offset, ci in enumerate(letter_indices):
                    mean_img = cluster_means[ci]

                    # Convert float [0, 1] → uint8 [0, 255], inverted
                    # (cluster means are bright-on-dark; app expects dark-on-white)
                    img_arr = mean_img
                    if img_arr.max() <= 1.0:
                        img_arr = (img_arr * 255).clip(0, 255)
                    img_arr = (255 - img_arr).astype(np.uint8)

                    # Sanitize letter for filesystem (e.g. '?' → '_QM_')
                    fname = f"{font_type}_{case_prefix}-{letter}_{idx_offset}.webp"
                    img = Image.fromarray(img_arr, mode='L')
                    img.save(str(book_out / fname), format='WEBP', quality=80, method=6)

                    all_letters_set.add(letter)
                    book_entries.append((font_type, letter))

        book_index[book_name] = sorted(set(book_entries))

    return book_index, sorted(all_letters_set)


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(
    corpus_dir: Path,
    config: dict,
    out_dir: Path,
    skip_images: bool = False,
    ocr_dir_override: Path | None = None,
) -> int:
    """Run the full export pipeline."""

    results_path = corpus_dir / "results" / "acontrario_results.npz"
    if not results_path.exists():
        # Also try without the results/ subdirectory
        results_path = corpus_dir / "acontrario_results.npz"
    if not results_path.exists():
        print(f"ERROR: No acontrario_results.npz found in {corpus_dir}",
              file=sys.stderr)
        return 1

    # ---- Load a contrario results ----
    print(f"\nLoading results from {results_path} …")
    cached = np.load(str(results_path), allow_pickle=True)
    books = cached["books"]
    printers = cached["printers"]

    n1hat_rm = cached.get("n1hat_roman")
    n_rm = cached.get("n_roman")
    n1hat_it = cached.get("n1hat_italic")
    n_it = cached.get("n_italic")

    n_books = len(books)
    print(f"  {n_books} books, "
          f"roman={'yes' if n1hat_rm is not None else 'no'}, "
          f"italic={'yes' if n1hat_it is not None else 'no'}")

    # ---- Compute ordering ----
    acfg = config["analysis"]
    metric_mode = acfg.get("metric", "average")
    ordering_mode = acfg.get("ordering", "hierarchical")
    print(f"  Ordering: {ordering_mode}, metric: {metric_mode}")

    order, w_rm, w_it = _compute_ordering(
        n1hat_rm, n_rm, n1hat_it, n_it, metric_mode, ordering_mode,
    )

    # ---- Reorder and save matrices ----
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_dir} …")

    if n1hat_rm is not None:
        np.save(str(out_dir / "n1hat_rm_matrix_ordered.npy"),
                _reorder_matrix(n1hat_rm, order).astype(np.float32))
        np.save(str(out_dir / "w_rm_matrix_ordered.npy"),
                _reorder_matrix(w_rm, order).astype(np.float32))
        print(f"  Saved n1hat_rm_matrix_ordered.npy ({n_books}×{n_books})")
        print(f"  Saved w_rm_matrix_ordered.npy")

    if n1hat_it is not None:
        np.save(str(out_dir / "n1hat_it_matrix_ordered.npy"),
                _reorder_matrix(n1hat_it, order).astype(np.float32))
        np.save(str(out_dir / "w_it_matrix_ordered.npy"),
                _reorder_matrix(w_it, order).astype(np.float32))
        print(f"  Saved n1hat_it_matrix_ordered.npy ({n_books}×{n_books})")
        print(f"  Saved w_it_matrix_ordered.npy")
        
    if n1hat_rm is not None and n1hat_it is not None:
        n1hat_combined = (n1hat_rm + n1hat_it)
        w_combined = (w_rm + w_it) / 2
        np.save(str(out_dir / "n1hat_combined_matrix_ordered.npy"),
                _reorder_matrix(n1hat_combined, order).astype(np.float32))
        np.save(str(out_dir / "w_combined_matrix_ordered.npy"),
                _reorder_matrix(w_combined, order).astype(np.float32))

    # Ordered book IDs and printer names
    np.save(str(out_dir / "books_dashboard_ordered.npy"), books[order])
    np.save(str(out_dir / "impr_names_dashboard_ordered.npy"), printers[order])
    print(f"  Saved books_dashboard_ordered.npy")
    print(f"  Saved impr_names_dashboard_ordered.npy")

    # ---- Symbols: union of letters from both distance files ----
    all_symbs: set[str] = set()
    for dist_name in ("distances_roman.npz", "distances_italic.npz"):
        dist_path = corpus_dir / dist_name
        if dist_path.exists():
            d = np.load(str(dist_path), allow_pickle=True)
            all_symbs.update(str(s) for s in d["letters"])
    np.save(str(out_dir / "symbs_dashboard.npy"),
            np.array(sorted(all_symbs)))
    print(f"  Saved symbs_dashboard.npy ({len(all_symbs)} symbols)")

    # ---- Glyph images ----
    if not skip_images:
        if ocr_dir_override:
            charnet_dir = ocr_dir_override
        else:
            # Prefer ocr/ layout, fall back to legacy charnet/
            ocr_root = corpus_dir / "ocr"
            if ocr_root.is_dir():
                engines = [d for d in sorted(ocr_root.iterdir()) if d.is_dir()]
                charnet_dir = engines[0] if len(engines) == 1 else corpus_dir / "charnet"
            else:
                charnet_dir = corpus_dir / "charnet"
        if not charnet_dir.exists():
            print(f"\n  Warning: No OCR output directory found at {charnet_dir}, "
                  "skipping image export")
        else:
            out_images_dir = out_dir / "images"
            out_images_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nExporting glyph images to {out_images_dir} …")
            t0 = time.time()

            book_index, all_letters = _export_glyph_images(
                charnet_dir, books, out_images_dir,
            )

            # Save images_cache_meta.pkl
            meta = {
                "all_letters": all_letters,
                "book_index": book_index,
            }
            meta_path = out_images_dir / "images_cache_meta.pkl"
            with open(str(meta_path), "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

            n_books_with_imgs = sum(1 for v in book_index.values() if v)
            n_total_imgs = sum(len(v) for v in book_index.values())
            elapsed = time.time() - t0
            print(f"  Exported images for {n_books_with_imgs}/{n_books} books "
                  f"({n_total_imgs} (font, letter) pairs) in {elapsed:.1f}s")
            print(f"  Saved images_cache_meta.pkl")

    print("\nDone.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a contrario results for the sueltas-app dashboard",
    )
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Path to the corpus directory (e.g. data/corpus-2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the a contrario YAML config (for ordering/metric mode)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <corpus_dir>/dashboard_export)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip glyph image export (matrices only)",
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=None,
        help="Path to OCR output directory with clustering results "
             "(default: auto-detect from ocr/ or charnet/)",
    )
    args = parser.parse_args(argv)

    corpus_dir = args.corpus_dir.resolve()
    if not corpus_dir.is_dir():
        print(f"ERROR: {corpus_dir} is not a directory", file=sys.stderr)
        return 1

    out_dir = args.out.resolve() if args.out else corpus_dir / "dashboard_export"
    config = _load_config(args.config)

    ocr_dir = args.ocr_dir.resolve() if args.ocr_dir else None
    return export(corpus_dir, config, out_dir,
                  skip_images=args.skip_images, ocr_dir_override=ocr_dir)


if __name__ == "__main__":
    sys.exit(main())
