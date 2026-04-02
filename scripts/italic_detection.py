#!/usr/bin/env python
"""
Italic detection for documents processed by character_extraction.py.

Per-document script that:
1. Loads all page .npz files (word_stroke_orientations, char_word_idx, char_labels)
2. Computes an Otsu-based stroke threshold across all words
3. Resolves ambiguous words using side-neighbor analysis
4. Outputs a single document-level .npz with per-character italic labels

Usage:
    python italic_detection.py data/corpus-1/charnet --output-dir data/corpus-1/italic
    python italic_detection.py data/corpus-1/charnet 
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from shared.tools.report import StepReport


# ---------------------------------------------------------------------------
# Ported from char-clust-trees/extract_chars/method.py
# ---------------------------------------------------------------------------

def get_complicated_bboxes(
    words: list[dict],
    strokes: np.ndarray,
    threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify 'complicated' (ambiguous) vs 'good' (clear) words for italic detection.

    A word is complicated if:
      - stroke == -360 (no stroke detected)
      - stroke is NaN
      - stroke >= threshold AND word contains z, y, or f (letters with diagonal strokes)

    Args:
        words: List of word dicts with 'tblr' and 'text' keys.
        strokes: Per-word stroke orientations, shape (n_words,).
        threshold: Stroke threshold for italic classification.

    Returns:
        bboxes: (n_words, 4) array [x_min, y_min, x_max, y_max]
        c_idxs: Indices of complicated words
        g_idxs: Indices of good words
    """
    bboxes = []
    c_idxs = []
    g_idxs = []

    for ix, word in enumerate(words):
        t, b, l, r = word["tblr"]
        bboxes.append([l, t, r, b])  # x_min, y_min, x_max, y_max

        stroke = strokes[ix]
        text = word.get("text", "").lower()

        is_complicated = (
            stroke == -360
            or np.isnan(stroke)
            or (
                stroke >= threshold
                and any(c in text for c in ("z", "y", "f"))
            )
        )

        if is_complicated:
            c_idxs.append(ix)
        else:
            g_idxs.append(ix)

    return (
        np.array(bboxes, dtype=np.int32),
        np.array(c_idxs, dtype=np.int64),
        np.array(g_idxs, dtype=np.int64),
    )


def find_side_neighbors(
    bboxes: np.ndarray,
    radius: float = 200,
    k: int = 2,
    y_tol: float = 0.5,
    side_tol: float = 10
) -> list[list[int]]:
    """
    Find horizontally adjacent word neighbors on the same text line.

    Args:
        bboxes: (n_words, 4) array [x_min, y_min, x_max, y_max]
        radius: Max distance for neighbor search (pixels).
        k: Max number of neighbors to return per word.
        y_tol: Vertical tolerance as fraction of word height.
        side_tol: Max distance between word edges (left-right adjacency).

    Returns:
        List of neighbor indices for each word, sorted by side distance.
    """
    if len(bboxes) == 0:
        return []

    centers = np.column_stack([
        (bboxes[:, 0] + bboxes[:, 2]) / 2,
        (bboxes[:, 1] + bboxes[:, 3]) / 2,
    ])
    heights = bboxes[:, 3] - bboxes[:, 1]
    tree = cKDTree(centers)

    neighbors = []
    for i, (bbox, h) in enumerate(zip(bboxes, heights)):
        idxs = tree.query_ball_point(centers[i], r=radius)
        # Filter: same line (vertical tolerance)
        idxs = [
            j for j in idxs
            if j != i and abs(centers[j][1] - centers[i][1]) < y_tol * h
        ]
        # Filter: left/right edges are close
        side_dists = [
            min(abs(bbox[2] - bboxes[j][0]), abs(bbox[0] - bboxes[j][2]))
            for j in idxs
        ]
        filtered = [j for j, d in zip(idxs, side_dists) if d < side_tol]
        # Sort by side distance, take top k
        filtered = sorted(
            filtered,
            key=lambda j: min(
                abs(bbox[2] - bboxes[j][0]),
                abs(bbox[0] - bboxes[j][2])
            ),
        )
        neighbors.append(filtered[:k])

    return neighbors


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_document(
    doc_dir: str | Path,
    output_dir: str | Path | None = None,
    skip_existing: bool = False,
) -> tuple[str, dict]:
    """
    Process a single document folder to compute italic labels.

    Returns (status_message, info_dict).
    """
    doc_dir = Path(doc_dir)
    doc_name = doc_dir.name
    info: dict = {"doc_name": doc_name}

    if output_dir is None:
        output_dir = doc_dir
    else:
        output_dir = Path(output_dir) / doc_name
        output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "italic_labels.npz"
    if skip_existing and out_path.exists():
        info["status"] = "skip"
        return f"SKIP: {doc_name} (already exists)", info

    # -- Step 1: Collect data from all pages ----------------------------------
    npz_files = sorted(doc_dir.glob("*_data.npz"))
    if not npz_files:
        info["status"] = "skip"
        return f"SKIP: {doc_name} (no .npz files)", info

    all_strokes = []
    page_data = []  # List of (words, strokes, char_word_idx, n_chars)

    for npz_path in npz_files:
        page_name = npz_path.stem.replace("_data", "")
        json_path = npz_path.parent / f"{page_name}.json"

        if not json_path.exists():
            continue

        with open(json_path, encoding="utf-8") as f:
            words = json.load(f)

        data = np.load(str(npz_path), allow_pickle=True)
        strokes = data["word_stroke_orientations"]
        char_word_idx = data.get("char_word_idx", np.array([], dtype=np.int32))

        all_strokes.append(strokes)
        page_data.append({
            "page_name": page_name,
            "words": words,
            "strokes": strokes,
            "char_word_idx": char_word_idx,
            "n_chars": len(char_word_idx),
        })

    if not page_data:
        info["status"] = "skip"
        return f"SKIP: {doc_name} (no valid pages)", info

    # -- Step 2: Compute threshold --------------------------------------------
    all_strokes_flat = np.concatenate(all_strokes)
    valid_strokes = all_strokes_flat[all_strokes_flat != -360]
    valid_strokes = valid_strokes[~np.isnan(valid_strokes)]

    if len(valid_strokes) < 10:
        info.update(status="skip", reason="too few valid strokes",
                    n_valid_strokes=int(len(valid_strokes)))
        return f"SKIP: {doc_name} (too few valid strokes: {len(valid_strokes)})", info

    q_85, q_90, q_95 = np.quantile(valid_strokes, [0.85, 0.9, 0.95])
    threshold = threshold_otsu(valid_strokes)

    # Sanity check: if Otsu is outside [q85, q95), use q90
    if not (q_85 <= threshold < q_95):
        threshold = q_90

    # -- Step 3: Assign italic labels per page --------------------------------
    all_char_italic = []
    page_char_counts = []

    for pd in page_data:
        words = pd["words"]
        strokes = pd["strokes"]
        char_word_idx = pd["char_word_idx"]
        n_chars = pd["n_chars"]

        if n_chars == 0:
            page_char_counts.append(0)
            continue

        # Get complicated vs good word indices
        bboxes, c_idxs, g_idxs = get_complicated_bboxes(words, strokes, threshold)

        # Build neighbor map for complicated words
        if len(c_idxs) > 0 and len(bboxes) > 0:
            neighbors = find_side_neighbors(bboxes, radius=200, k=2, y_tol=0.5, side_tol=10)
        else:
            neighbors = [[] for _ in range(len(words))]

        # Convert g_idxs to set for fast lookup
        g_idxs_set = set(g_idxs.tolist())

        # Compute effective stroke per word (resolving ambiguous via neighbors)
        word_effective_stroke = strokes.copy()
        for ix in c_idxs:
            neigh_ixs = neighbors[ix]
            n_strokes = [strokes[nei] for nei in neigh_ixs if nei in g_idxs_set]
            if n_strokes:
                word_effective_stroke[ix] = np.max(n_strokes)

        # Assign italic label per character
        char_italic = np.zeros(n_chars, dtype=np.int8)
        for ci, wi in enumerate(char_word_idx):
            if wi < len(word_effective_stroke):
                char_italic[ci] = 1 if word_effective_stroke[wi] >= threshold else 0

        all_char_italic.append(char_italic)
        page_char_counts.append(n_chars)

    # -- Step 4: Save results -------------------------------------------------
    char_italic_flat = np.concatenate(all_char_italic) if all_char_italic else np.array([], dtype=np.int8)
    page_char_counts = np.array(page_char_counts, dtype=np.int32)
    page_names = [pd["page_name"] for pd in page_data]

    np.savez_compressed(
        str(out_path),
        char_italic=char_italic_flat,
        page_char_counts=page_char_counts,
        page_names=np.array(page_names, dtype="U64"),
        threshold=np.array([threshold]),
        n_valid_strokes=np.array([len(valid_strokes)]),
    )

    n_italic = char_italic_flat.sum()
    n_total = len(char_italic_flat)
    pct = 100 * n_italic / n_total if n_total > 0 else 0

    info.update(
        status="ok",
        n_pages=len(page_data),
        total_characters=int(n_total),
        n_roman=int(n_total - n_italic),
        n_italic=int(n_italic),
        italic_ratio=round(pct / 100, 4),
        otsu_threshold=round(float(threshold), 4),
        n_valid_strokes=int(len(valid_strokes)),
        quantiles={"q85": round(float(q_85), 4),
                   "q90": round(float(q_90), 4),
                   "q95": round(float(q_95), 4)},
    )

    return (
        f"OK: {doc_name}  "
        f"({len(page_data)} pages, {n_total} chars, "
        f"{n_italic} italic [{pct:.1f}%], threshold={threshold:.2f})",
        info,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute italic labels for documents processed by character_extraction.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/italic_detection.py data/corpus-1/charnet 
  python scripts/italic_detection.py data/corpus-1/charnet/doc001 --single-doc
        """,
    )
    parser.add_argument(
        "input_dir",
        help="Document folder with *_data.npz files, "
             "or parent folder if --process-subfolders is set or --single-doc",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for italic_labels.npz files. "
             "Defaults to same as input.",
    )
    # Subfolder processing is now default unless --single-doc is set
    parser.add_argument(
        "--single-doc",
        action="store_true",
        help="Process a single document folder (cohesive with previous steps)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have italic_labels.npz",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for multi-document mode (default: ncpus-1, 0 = auto)",
    )
    parser.add_argument(
        "--report-dir",
        help="Directory for the step report JSON (default: auto)",
    )
    args = parser.parse_args(argv)

    start = time.time()
    num_workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    report = StepReport("italic_detection")

    if args.single_doc:
        result, info = process_document(
            args.input_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
        )
        report.add_document(info.get("doc_name", "unknown"), info)
        print(result)
    else:
        # Default: process each subfolder as a separate document
        subfolders = sorted(
            Path(args.input_dir) / d
            for d in os.listdir(args.input_dir)
            if (Path(args.input_dir) / d).is_dir()
        )
        n = len(subfolders)
        print(f"Processing {n} documents with {num_workers} workers...")

        if num_workers == 1:
            for i, subfolder in enumerate(subfolders, 1):
                result, info = process_document(
                    subfolder,
                    output_dir=args.output_dir,
                    skip_existing=args.skip_existing,
                )
                report.add_document(info.get("doc_name", subfolder.name), info)
                print(f"[{i}/{n}] {result}")
        else:
            done = 0
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(
                        process_document,
                        subfolder,
                        output_dir=args.output_dir,
                        skip_existing=args.skip_existing,
                    ): subfolder
                    for subfolder in subfolders
                }
                for fut in as_completed(futures):
                    done += 1
                    result, info = fut.result()
                    report.add_document(info.get("doc_name", futures[fut].name), info)
                    print(f"[{done}/{n}] {result}")

    # -- Summary & report ----------------------------------------------------
    total_chars = sum(d.get("total_characters", 0) for d in report.documents.values())
    total_italic = sum(d.get("n_italic", 0) for d in report.documents.values())
    n_ok = sum(1 for d in report.documents.values() if d.get("status") == "ok")
    report.set_summary({
        "total_documents": len(report.documents),
        "documents_processed": n_ok,
        "total_characters": total_chars,
        "total_italic": total_italic,
        "overall_italic_ratio": round(total_italic / total_chars, 4) if total_chars else 0,
    })

    input_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir) if args.report_dir else (
        (input_dir if args.single_doc else input_dir.parent) / "reports")
    rpath = report.save(report_dir)
    print(f"Report saved: {rpath}")

    print(f"\nTotal time: {time.time() - start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
