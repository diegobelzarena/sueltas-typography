#!/usr/bin/env python
"""
Character clustering for documents processed by character_extraction.py.

Per-document script that:
1. Loads all page .npz files (char_imgs, char_labels, char_word_idx)
2. Loads italic labels from italic_detection.py output
3. Filters invalid images (NaN, low-information)
4. Runs GMM clustering + tree refinement
5. Computes cluster statistics (means, majority labels, italic ratios)
6. Saves results to clusters_all.npz

Usage:
    python clustering.py data/corpus-1/charnet/BNE_1001_615_T-55281-18
    python clustering.py data/corpus-1/charnet --process-subfolders
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable even without pip install -e .
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from image_processing.clustering import clusterize_gmm, tree_refine
from image_processing.tools.inverse_compositional import register2mean
from shared.tools.report import StepReport


def process_document(
    doc_dir: str | Path,
    n_jobs: int = 1,
    skip_existing: bool = False,
    device: str = "cpu",
) -> tuple[str, dict]:
    """
    Process a single document folder to cluster characters.

    Returns (status_message, info_dict).
    """
    doc_dir = Path(doc_dir)
    doc_name = doc_dir.name
    info: dict = {"doc_name": doc_name}

    out_path = doc_dir / "clusters_all.npz"
    if skip_existing and out_path.exists():
        info["status"] = "skip"
        return f"SKIP: {doc_name} (clusters already exist)", info

    print(f"\n{'='*60}")
    print(f"Processing: {doc_name}")
    print(f"{'='*60}")

    # -- Step 1: Load all character data from pages ---------------------------
    print("\nStep 1/5: Loading character data from pages...")
    step_start = time.time()

    npz_files = sorted(doc_dir.glob("*_data.npz"))
    if not npz_files:
        return f"SKIP: {doc_name} (no .npz files)"

    all_char_imgs = []
    all_char_labels = []
    all_word_idx = []  # Global word index across all pages
    page_heights = []
    total_words = 0

    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)

        if "char_imgs" not in data or len(data["char_imgs"]) == 0:
            continue

        char_imgs = data["char_imgs"]
        char_labels = data["char_labels"]
        char_word_idx = data["char_word_idx"]
        page_height = int(data["page_height"][0])

        # Offset word indices to be global
        global_word_idx = char_word_idx + total_words
        n_words_page = int(char_word_idx.max()) + 1 if len(char_word_idx) > 0 else 0
        total_words += n_words_page

        all_char_imgs.append(char_imgs)
        all_char_labels.append(char_labels)
        all_word_idx.append(global_word_idx)
        page_heights.append(page_height)

    if not all_char_imgs:
        info["status"] = "skip"
        return f"SKIP: {doc_name} (no character images found)", info

    imgs = np.concatenate(all_char_imgs)
    labels = np.concatenate(all_char_labels)
    word_idx = np.concatenate(all_word_idx)

    print(f"  Found {len(imgs)} characters from {len(npz_files)} pages")
    print(f"  Time: {time.time() - step_start:.2f}s")

    # -- Step 2: Load italic labels -------------------------------------------
    print("\nStep 2/5: Loading italic labels...")
    step_start = time.time()

    italic_path = doc_dir / "italic_labels.npz"
    if italic_path.exists():
        italic_data = np.load(str(italic_path), allow_pickle=True)
        italic_labels = italic_data["char_italic"]
        if len(italic_labels) != len(imgs):
            print(f"  Warning: italic label count mismatch ({len(italic_labels)} vs {len(imgs)})")
            italic_labels = np.zeros(len(imgs), dtype=np.int8)
    else:
        print("  Warning: No italic labels found, setting all to 0")
        italic_labels = np.zeros(len(imgs), dtype=np.int8)

    print(f"  Italic: {italic_labels.sum()} ({100*italic_labels.mean():.1f}%)")
    print(f"  Time: {time.time() - step_start:.2f}s")

    # -- Step 3: Filter invalid images ----------------------------------------
    print("\nStep 3/5: Filtering invalid images...")
    step_start = time.time()

    # Filter NaN values
    valid_mask = ~np.isnan(imgs).any(axis=(1, 2))
    if not valid_mask.all():
        n_nan = (~valid_mask).sum()
        print(f"  Removing {n_nan} images with NaN values")

    # Filter low-information images
    # Note: images from embed_noresize are inverted (background≈0, text≈1)
    # and interpolated, so use near-zero threshold instead of exact equality
    img_stds = imgs[valid_mask].std(axis=(1, 2))
    near_zero_ratios = (imgs[valid_mask] < 0.05).sum(axis=(1, 2)) / (imgs.shape[1] * imgs.shape[2])

    # Diagnostic: show distribution of filter values
    print(f"  Std range: [{img_stds.min():.4f}, {img_stds.max():.4f}], "
          f"median: {np.median(img_stds):.4f}")
    print(f"  Near-zero ratio range: [{near_zero_ratios.min():.4f}, {near_zero_ratios.max():.4f}], "
          f"median: {np.median(near_zero_ratios):.4f}")

    std_pass = (img_stds > 0.01).sum()
    ratio_pass = (near_zero_ratios < 0.99).sum()
    print(f"  Images passing std>0.01: {std_pass}/{len(img_stds)}")
    print(f"  Images passing ratio<0.99: {ratio_pass}/{len(near_zero_ratios)}")

    info_mask_sub = (img_stds > 0.01) & (near_zero_ratios < 0.99)

    # Combine masks
    valid_indices = np.where(valid_mask)[0]
    final_valid = valid_indices[info_mask_sub]

    n_removed = len(imgs) - len(final_valid)
    if n_removed > 0:
        print(f"  Removing {n_removed} low-information images")

    imgs_filtered = imgs[final_valid]
    labels_filtered = labels[final_valid]
    italic_filtered = italic_labels[final_valid]
    word_idx_filtered = word_idx[final_valid]

    if len(imgs_filtered) == 0:
        info["status"] = "skip"
        return f"SKIP: {doc_name} (no valid images after filtering)", info

    print(f"  Valid images: {len(imgs_filtered)}")
    print(f"  Time: {time.time() - step_start:.2f}s")

    # -- Step 4: GMM clustering + tree refinement -----------------------------
    print("\nStep 4/5: Clustering...")
    step_start = time.time()

    try:
        print("  Running initial GMM clustering...")
        clu_pred = clusterize_gmm(
            imgs_filtered, pca_level=0.9, n_comps=70, seed=42, device=device
        )
        print(f"  Found {len(np.unique(clu_pred))} initial clusters")

        print("  Refining with tree-based splitting...")
        clu_pred_refined = tree_refine(
            imgs_filtered,
            clu_pred,
            transform="euclidean",
            min_imgs=20,
            pca_level=9,
            num_tests=5,
            test="ad",
            seed=0,
            n_jobs=n_jobs,
            max_depth=10,
            skip_registration=True,
        )
        print(f"  Refined to {len(np.unique(clu_pred_refined[clu_pred_refined >= 0]))} final clusters")

    except (np.linalg.LinAlgError, RuntimeError) as e:
        info.update(status="error", error=str(e))
        return f"ERROR: {doc_name} - Clustering failed: {e}", info

    print(f"  Time: {time.time() - step_start:.2f}s")

    # -- Step 5: Compute cluster statistics -----------------------------------
    print("\nStep 5/5: Computing cluster statistics...")
    step_start = time.time()

    clusters = np.unique(clu_pred_refined[clu_pred_refined >= 0])
    cluster_masks_dict = {c: (clu_pred_refined == c) for c in clusters}

    cluster_means = []
    cluster_labels_list = []
    cluster_italic_list = []

    for cluster in tqdm(clusters, desc="  Computing cluster means"):
        mask = cluster_masks_dict[cluster]
        cluster_labels_arr = labels_filtered[mask]

        # Find majority label
        labs, counts = np.unique(cluster_labels_arr, return_counts=True)
        majority_lab = labs[np.argmax(counts)]

        # Skip punctuation clusters
        if majority_lab in {".", ",", ";", ":"}:
            continue

        majority_pct = np.max(counts) / np.sum(counts)
        if majority_pct < 0.6:
            continue

        cluster_imgs = imgs_filtered[mask]

        # Compute mean (with optional registration for small clusters)
        if cluster_imgs.shape[0] < 400:
            try:
                cluster_imgs_reg = register2mean(cluster_imgs, transform="translation")
                cluster_mean = cluster_imgs_reg.mean(axis=0)
            except Exception:
                cluster_mean = cluster_imgs.mean(axis=0)
        else:
            cluster_mean = cluster_imgs.mean(axis=0)

        cluster_means.append(cluster_mean)
        cluster_labels_list.append([majority_lab, majority_pct, len(cluster_imgs)])
        cluster_italic_list.append(italic_filtered[mask].mean())

    cluster_means = np.array(cluster_means) if cluster_means else np.zeros((0, 40, 32))
    cluster_labels_arr = np.array(cluster_labels_list, dtype=object) if cluster_labels_list else np.array([])
    cluster_italic_arr = np.array(cluster_italic_list) if cluster_italic_list else np.array([])

    print(f"  Final valid clusters: {len(cluster_means)}")
    print(f"  Time: {time.time() - step_start:.2f}s")

    # -- Save results ---------------------------------------------------------
    print("\nSaving results...")
    np.savez(
        str(out_path),
        cluster_means=cluster_means,
        cluster_labels=cluster_labels_arr,
        cluster_italic=cluster_italic_arr,
    )

    # Compute report stats
    n_roman_clusters = int((cluster_italic_arr <= 0.1).sum()) if len(cluster_italic_arr) else 0
    n_italic_clusters = int((cluster_italic_arr >= 0.4).sum()) if len(cluster_italic_arr) else 0
    n_ambiguous_clusters = int(len(cluster_italic_arr) - n_roman_clusters - n_italic_clusters)
    # Top letters by sample count
    top_letters = {}
    for cl in cluster_labels_list:
        letter, conf, count = cl[0], cl[1], cl[2]
        top_letters[letter] = top_letters.get(letter, 0) + count
    top_letters_sorted = sorted(top_letters.items(), key=lambda x: -x[1])[:10]

    info.update(
        status="ok",
        n_pages=len(npz_files),
        total_characters_input=int(len(imgs)),
        n_invalid_removed=n_removed,
        n_characters_clustered=int(len(imgs_filtered)),
        n_clusters_initial=int(len(np.unique(clu_pred))),
        n_clusters_final=int(len(cluster_means)),
        n_roman_clusters=n_roman_clusters,
        n_italic_clusters=n_italic_clusters,
        n_ambiguous_clusters=n_ambiguous_clusters,
        mean_cluster_confidence=round(float(np.mean(
            [float(cl[1]) for cl in cluster_labels_list])), 4) if cluster_labels_list else 0,
        top_letters=top_letters_sorted,
    )

    return (
        f"OK: {doc_name} \u2014 "
        f"{len(imgs_filtered)} chars, "
        f"{len(cluster_means)} clusters",
        info,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cluster characters from processed document pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/clustering.py data/corpus-1/charnet --workers 4
  python scripts/clustering.py data/corpus-1/charnet/doc001 --single-doc
        """,
    )
    parser.add_argument(
        "input_dir",
        help="Document folder with *_data.npz files, "
             "or parent folder (processes subfolders by default)",
    )
    parser.add_argument(
        "--single-doc",
        action="store_true",
        help="Process a single document folder (cohesive with previous steps)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have clusters_all.npz",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for tree refinement",
    )
    parser.add_argument(
        "--report-dir",
        help="Directory for the step report JSON (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device ('cpu' or 'cuda')",
    )
    args = parser.parse_args(argv)

    start = time.time()
    report = StepReport("clustering")

    if args.single_doc:
        result, info = process_document(
            args.input_dir,
            n_jobs=args.workers,
            skip_existing=args.skip_existing,
            device=args.device,
        )
        report.add_document(info.get("doc_name", "unknown"), info)
        print(f"\n{result}")
    else:
        # Default: process each subfolder as a separate document
        subfolders = sorted(
            Path(args.input_dir) / d
            for d in os.listdir(args.input_dir)
            if (Path(args.input_dir) / d).is_dir()
        )
        print(f"Processing {len(subfolders)} documents...")
        for i, subfolder in enumerate(subfolders, 1):
            result, info = process_document(
                subfolder,
                n_jobs=args.workers,
                skip_existing=args.skip_existing,
                device=args.device,
            )
            report.add_document(info.get("doc_name", subfolder.name), info)
            print(f"\n[{i}/{len(subfolders)}] {result}")

    # -- Summary & report ----------------------------------------------------
    total_input = sum(d.get("total_characters_input", 0) for d in report.documents.values())
    total_clustered = sum(d.get("n_characters_clustered", 0) for d in report.documents.values())
    total_clusters = sum(d.get("n_clusters_final", 0) for d in report.documents.values())
    n_ok = sum(1 for d in report.documents.values() if d.get("status") == "ok")
    report.set_summary({
        "total_documents": len(report.documents),
        "documents_processed": n_ok,
        "total_characters_input": total_input,
        "total_characters_clustered": total_clustered,
        "total_clusters": total_clusters,
    })

    input_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir) if args.report_dir else (
        (input_dir if args.single_doc else input_dir.parent) / "reports")
    rpath = report.save(report_dir)
    print(f"Report saved: {rpath}")

    print(f"\n{'='*60}")
    print(f"Total time: {time.time() - start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
