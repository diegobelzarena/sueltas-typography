#!/usr/bin/env python3
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_processing.clustering import clusterize_gmm, tree_refine
from image_processing.tools.inverse_compositional import register2mean


def process_document(
    doc_dir: str | Path,
    n_jobs: int = 1,
    skip_existing: bool = False,
    device: str = "cpu",
) -> str:
    """
    Process a single document folder to cluster characters.

    Args:
        doc_dir: Path to document folder with *_data.npz files.
        n_jobs: Number of parallel jobs for tree refinement.
        skip_existing: Skip if clusters_all.npz already exists.
        device: PyTorch device ('cpu' or 'cuda').

    Returns:
        Status message.
    """
    doc_dir = Path(doc_dir)
    doc_name = doc_dir.name

    out_path = doc_dir / "clusters_all.npz"
    if skip_existing and out_path.exists():
        return f"SKIP: {doc_name} (clusters already exist)"

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
        return f"SKIP: {doc_name} (no character images found)"

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
        return f"SKIP: {doc_name} (no valid images after filtering)"

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
            transform="euclidian",
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
        return f"ERROR: {doc_name} - Clustering failed: {e}"

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

    return (
        f"OK: {doc_name} — "
        f"{len(imgs_filtered)} chars, "
        f"{len(cluster_means)} clusters"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cluster characters from processed document pages"
    )
    parser.add_argument(
        "input_dir",
        help="Document folder with *_data.npz files, "
             "or parent folder if --process-subfolders is set",
    )
    parser.add_argument(
        "--process-subfolders",
        action="store_true",
        help="Process each subfolder as a separate document",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have clusters_all.npz",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for tree refinement",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device ('cpu' or 'cuda')",
    )
    args = parser.parse_args(argv)

    start = time.time()

    if args.process_subfolders:
        subfolders = sorted(
            Path(args.input_dir) / d
            for d in os.listdir(args.input_dir)
            if (Path(args.input_dir) / d).is_dir()
        )
        print(f"Processing {len(subfolders)} documents...")
        for i, subfolder in enumerate(subfolders, 1):
            result = process_document(
                subfolder,
                n_jobs=args.n_jobs,
                skip_existing=args.skip_existing,
                device=args.device,
            )
            print(f"\n[{i}/{len(subfolders)}] {result}")
    else:
        result = process_document(
            args.input_dir,
            n_jobs=args.n_jobs,
            skip_existing=args.skip_existing,
            device=args.device,
        )
        print(f"\n{result}")

    print(f"\n{'='*60}")
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
