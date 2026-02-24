#!/usr/bin/env python
"""Compute typographic distances between documents based on cluster means.

This script computes inter-document distances by comparing character cluster
centroids. It processes roman and italic styles separately.

For each style, the algorithm:
1. Loads cluster means from all documents
2. Filters clusters by italic ratio and label confidence
3. Keeps top N clusters per letter per document
4. Registers images to their mean using inverse compositional
5. Computes cosine distances between cluster means per letter
6. Aggregates per-letter distances into document-level matrices

Usage
-----
    python scripts/typographic_distances.py data/corpus-1

    # Process only roman
    python scripts/typographic_distances.py data/corpus-1 --style roman

    # Use custom config
    python scripts/typographic_distances.py data/corpus-1 --config my_config.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from image_processing.tools.inverse_compositional import register2mean


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "roman": {"italic_max": 0.1},
    "italic": {"italic_min": 0.4},
    "filtering": {
        "label_confidence_min": 0.6,
        "top_clusters_per_letter": 5,
        "min_doc_coverage": 0.33,
        "excluded_chars": [",", ".", ";", ":"],
    },
    "registration": {"transform": "euclidian"},
    "distance": {"metric": "cosine"},
}


def load_config(config_path: Path | None) -> dict:
    """Load configuration from YAML file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        if user_config:
            # Deep merge
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
    return config


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata(corpus_dir: Path) -> pd.DataFrame | None:
    """Load metadata CSV from corpus directory."""
    # Look for CSV files in corpus root
    csv_files = list(corpus_dir.glob("*.csv"))
    if not csv_files:
        return None
    
    # Prefer files with "table" or "corpus" in name
    for pattern in ["*table*", "*corpus*", "*meta*"]:
        matches = list(corpus_dir.glob(pattern + ".csv"))
        if matches:
            csv_files = matches
            break
    
    csv_path = csv_files[0]
    print(f"  Loading metadata from: {csv_path.name}")
    
    try:
        df = pd.read_csv(csv_path)
        doc_col = "Document"
        printer_col = "Printer"
        
        if doc_col not in df.columns:
            print(f"  Warning: '{doc_col}' column not found in metadata")
            return None
        
        return df
    except Exception as e:
        print(f"  Warning: Could not load metadata: {e}")
        return None


def get_printer_name(doc_name: str, metadata: pd.DataFrame | None) -> str:
    """Get printer name for a document from metadata."""
    if metadata is None:
        return "unknown"
    
    match = metadata[metadata["FileName"] == doc_name]
    if match.empty:
        # Try partial match
        for idx, row in metadata.iterrows():
            if doc_name in str(row["FileName"]) or str(row["FileName"]) in doc_name:
                printer = row["Printer"]
                return str(printer) if pd.notna(printer) else "unknown"
        return "unknown"
    
    printer = match["Printer"].values[0]
    return str(printer) if pd.notna(printer) else "unknown"


# ---------------------------------------------------------------------------
# Per-document distance aggregation
# ---------------------------------------------------------------------------

def per_doc_inf(distances: np.ndarray, y_names: np.ndarray):
    """
    Aggregate pairwise distances to per-document minimum distances.
    
    Args:
        distances: Pairwise distance matrix (n_samples, n_samples)
        y_names: Document name for each sample
    
    Returns:
        agg_distances: (n_docs, n_docs) minimum distances between documents
        arg_mins: (n_docs, n_docs, 2) indices of minimum distance pairs
    """
    distances_df = pd.DataFrame(distances, index=y_names, columns=y_names)
    
    idx = np.unique(y_names, return_index=True)[1]
    names = np.array([y_names[index] for index in sorted(idx)])
    n = len(names)
    
    agg_distances = np.zeros((n, n))
    arg_mins = np.zeros((n, n, 2), dtype=int)
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            dist = distances_df.loc[[names[i]], [names[j]]].values
            agg_distances[i, j] = dist.min(1).min()
            agg_distances[j, i] = dist.min(0).min()
            arg_mins[i, j] = np.unravel_index(dist.argmin(), dist.shape)
    
    return agg_distances, arg_mins


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def load_clusters_for_style(
    charnet_dir: Path,
    style: str,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and filter cluster data for a specific style (roman/italic).
    
    Returns:
        mean_imgs: Cluster mean images
        labels: Majority OCR label per cluster
        names: Document name per cluster
        folders: List of all document names
    """
    filtering = config["filtering"]
    excluded = set(filtering["excluded_chars"])
    label_conf_min = filtering["label_confidence_min"]
    top_n = filtering["top_clusters_per_letter"]
    
    if style == "roman":
        italic_max = config["roman"]["italic_max"]
        italic_min = None
    else:
        italic_max = None
        italic_min = config["italic"]["italic_min"]
    
    mean_imgs = []
    labels = []
    names = []
    folders = []
    
    # Find all documents with clustering results
    for doc_dir in sorted(charnet_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        
        cluster_path = doc_dir / "clusters_all.npz"
        if not cluster_path.exists():
            continue
        
        folders.append(doc_dir.name)
        
        try:
            data = np.load(str(cluster_path), allow_pickle=True)
        except Exception as e:
            print(f"  Warning: Could not load {cluster_path}: {e}")
            continue
        
        cluster_means = data["cluster_means"]
        cluster_labels = data["cluster_labels"]
        cluster_italic = data["cluster_italic"]
        
        if len(cluster_italic) == 0:
            continue
        
        # Filter by style (italic ratio)
        if style == "roman":
            style_mask = cluster_italic <= italic_max
        else:
            style_mask = cluster_italic >= italic_min
        
        # Filter by label confidence
        label_mask = np.array([
            float(cl[1]) >= label_conf_min 
            for cl in cluster_labels
        ])
        
        # Combined initial mask
        init_mask = style_mask & label_mask
        
        if not init_mask.any():
            continue
        
        # Get selected labels
        selected_labels = cluster_labels[init_mask][:, 0]
        
        # Keep top N per letter by sample count
        unique_letters = np.unique(selected_labels)
        final_indices = []
        init_indices = np.where(init_mask)[0]
        
        for letter in unique_letters:
            if letter in excluded:
                continue
            
            letter_mask = selected_labels == letter
            letter_indices = init_indices[letter_mask]
            
            if len(letter_indices) > top_n:
                # Sort by sample count and keep top N
                counts = [int(cluster_labels[i][2]) for i in letter_indices]
                top_idx = np.argsort(counts)[-top_n:]
                letter_indices = letter_indices[top_idx]
            
            final_indices.extend(letter_indices)
        
        if not final_indices:
            continue
        
        # Collect data
        for idx in final_indices:
            mean_imgs.append(cluster_means[idx])
            labels.append(cluster_labels[idx][0])
            names.append(doc_dir.name)
    
    return (
        np.array(mean_imgs) if mean_imgs else np.array([]),
        np.array(labels) if labels else np.array([]),
        np.array(names) if names else np.array([]),
        folders,
    )


def compute_distances(
    mean_imgs: np.ndarray,
    labels: np.ndarray,
    names: np.ndarray,
    folders: list[str],
    config: dict,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-letter adjacency matrices and aggregate.
    
    Returns:
        Adj: (n_letters, n_docs, n_docs) adjacency matrices per letter
        letters: List of letters included
    """
    filtering = config["filtering"]
    excluded = set(filtering["excluded_chars"])
    min_coverage = filtering["min_doc_coverage"]
    transform = config["registration"]["transform"]
    metric = config["distance"]["metric"]
    
    # Get unique letters
    all_letters = np.unique(labels)
    all_letters = [l for l in all_letters if l not in excluded]
    
    n_docs = len(folders)
    Adj = np.zeros((len(all_letters), n_docs, n_docs))
    valid_letters = []
    
    for l_idx, letter in enumerate(tqdm(all_letters, desc="  Processing letters")):
        mask = labels == letter
        uq_names = np.unique(names[mask])
        
        # Check minimum document coverage
        if len(uq_names) < (n_docs * min_coverage):
            continue
        
        valid_letters.append(letter)
        
        # Get images for this letter
        l_imgs = mean_imgs[mask].copy()
        l_names = names[mask]
        
        # Register to mean
        try:
            tf_means = register2mean(l_imgs, transform=transform)
        except Exception:
            tf_means = l_imgs
        
        # Compute pairwise distances
        p_dist = pairwise_distances(
            tf_means.reshape(len(tf_means), -1), 
            metric=metric
        )
        
        # Aggregate to per-document distances
        distances_cos, _ = per_doc_inf(p_dist, l_names)
        distances_cos = (distances_cos + distances_cos.T) / 2
        
        # Fill adjacency matrix
        # Initialize with max distance for missing pairs
        Adj[l_idx] += 2 * distances_cos.max()
        
        for k, nm_0 in enumerate(uq_names):
            i = np.where(np.array(folders) == nm_0)[0]
            if len(i) == 0:
                continue
            i = i[0]
            
            for m, nm_1 in enumerate(uq_names[k + 1:]):
                j = np.where(np.array(folders) == nm_1)[0]
                if len(j) == 0:
                    continue
                j = j[0]
                
                Adj[l_idx, i, j] += distances_cos[k, m + k + 1] - (2 * distances_cos.max())
                Adj[l_idx, j, i] += distances_cos[k, m + k + 1] - (2 * distances_cos.max())
    
    # Remove letters with all zeros (no valid comparisons)
    row_sums = Adj.sum(axis=(1, 2))
    nonzero_rows = row_sums != 0
    Adj = Adj[nonzero_rows]
    valid_letters = [valid_letters[i] for i in range(len(valid_letters)) if nonzero_rows[i]]
    
    return Adj, valid_letters


def process_style(
    corpus_dir: Path,
    charnet_dir: Path,
    style: str,
    metadata: pd.DataFrame | None,
    config: dict,
) -> str:
    """Process a single style (roman or italic)."""
    print(f"\n{'='*60}")
    print(f"Processing style: {style}")
    print(f"{'='*60}")
    
    # Load cluster data
    print("\nStep 1/3: Loading cluster data...")
    mean_imgs, labels, names, folders = load_clusters_for_style(
        charnet_dir, style, config
    )
    
    if len(mean_imgs) == 0:
        return f"SKIP: No valid clusters for style '{style}'"
    
    print(f"  Documents: {len(folders)}")
    print(f"  Clusters: {len(mean_imgs)}")
    print(f"  Unique letters: {len(np.unique(labels))}")
    
    # Get printer names
    print("\nStep 2/3: Mapping metadata...")
    printer_names = np.array([
        get_printer_name(folder, metadata, config)
        for folder in folders
    ])
    
    n_known = sum(1 for p in printer_names if p != "unknown")
    print(f"  Documents with printer info: {n_known}/{len(folders)}")
    
    # Compute distances
    print("\nStep 3/3: Computing distances...")
    Adj, letters = compute_distances(mean_imgs, labels, names, folders, config)
    
    if len(Adj) == 0:
        return f"SKIP: No valid letter comparisons for style '{style}'"
    
    print(f"  Valid letters: {len(letters)}")
    print(f"  Adjacency shape: {Adj.shape}")
    
    # Save results
    out_path = corpus_dir / f"distances_{style}.npz"
    np.savez(
        str(out_path),
        doc_names=np.array(folders),
        printer_names=printer_names,
        adjacencies=Adj,
        letters=np.array(letters),
    )
    
    print(f"\n  Saved: {out_path}")
    return f"OK: {style} — {len(folders)} docs, {len(letters)} letters"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute typographic distances between documents"
    )
    parser.add_argument(
        "corpus_dir",
        help="Corpus directory (containing charnet/ subfolder with clustering results)",
    )
    parser.add_argument(
        "--style",
        choices=["roman", "italic", "both"],
        default="both",
        help="Which style to process (default: both)",
    )
    parser.add_argument(
        "--config",
        help="Path to config YAML file (default: configs/typographic_distances.yaml)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip styles that already have distance files",
    )
    args = parser.parse_args(argv)
    
    corpus_dir = Path(args.corpus_dir).resolve()
    charnet_dir = corpus_dir / "charnet"
    
    # Validate paths
    if not corpus_dir.is_dir():
        print(f"ERROR: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 1
    
    if not charnet_dir.is_dir():
        print(f"ERROR: CharNet output not found: {charnet_dir}", file=sys.stderr)
        return 1
    
    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent.parent / "configs" / "typographic_distances.yaml"
    
    config = load_config(config_path)
    print(f"Config: {config_path if config_path.exists() else 'defaults'}")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = load_metadata(corpus_dir)
    
    # Determine styles to process
    styles = ["roman", "italic"] if args.style == "both" else [args.style]
    
    # Process each style
    start = time.time()
    results = []
    
    for style in styles:
        out_path = corpus_dir / f"distances_{style}.npz"
        if args.skip_existing and out_path.exists():
            print(f"\nSkipping {style} (already exists)")
            results.append(f"SKIP: {style} (already exists)")
            continue
        
        result = process_style(corpus_dir, charnet_dir, style, metadata, config)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r}")
    print(f"\nTotal time: {time.time() - start:.1f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
