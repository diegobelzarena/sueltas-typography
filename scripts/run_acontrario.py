#!/usr/bin/env python
"""A contrario analysis and visualisation of typographic distances.

Loads pre‑computed distance matrices (``distances_roman.npz``,
``distances_italic.npz``), runs the a contrario detection, and produces:
  • *n̂₁* matrix heatmaps (one per font style),
  • weighted UMAP graphs (one per font style),
  • intermediate results saved as NPZ (``acontrario_results.npz``).

Usage
-----
    # Run with the corpus‑1 config
    python scripts/run_acontrario.py data/corpus-1 --config configs/acontrario_corpus1.yaml

    # Process only the graph (skip matrices)
    python scripts/run_acontrario.py data/corpus-1 --config configs/acontrario_corpus1.yaml --plots graph

    # Quick re‑run reusing saved intermediate results
    python scripts/run_acontrario.py data/corpus-1 --config configs/acontrario_corpus1.yaml --load-results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from acontrario import (
    build_alpha_grid,
    estimate_all_quantiles_,
    estimate_quantiles_cross_corpus,
    acontrario as run_acontrario,
    hierarchical_olo_order,
    load_adjacencies,
)
from acontrario.visualization import plot_graph, plot_matrix
from shared.tools.report import StepReport


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "analysis": {
        "epsilon": 0.01,
        "alphas": {
            "base_factors": [0.1, 0.15],
            "exponent_range": [-5, 3],
        },
        # Which weight matrix to use for the hierarchical ordering & graph
        # distances: "average" (roman+italic)/2, "roman", or "italic".
        # Falls back gracefully when the chosen style is unavailable.
        "ordering": "average",
        # Optional path to a reference corpus whose distances are used to
        # estimate the background quantiles (cross-corpus calibration).
        # Letters shared between both corpora use the reference; letters
        # only in the target corpus use the target's own distances.
        # Set to null / omit to estimate quantiles from the target itself.
        "reference_corpus": None,
    },
    "printers": {
        "colors": {},
        "short_name": "last_word",
    },
    "remarks": {
        "shapes": {
            "known": "D",
            "hidden": "h",
            "new": "*",
            "nan": "s",
        },
    },
    "visualization": {
        "matrix": {
            "figsize_per_book": 0.2,
            "include_values": False,
            "cmap": "viridis",
            "xticks_rotation": "vertical",
            "colorbar": True,
            "colorbar_step": 5,
            "marker_edge_color": "white",
            "marker_edge_width": 0.4,
            "marker_sizes": {},
            "legend_marker_size": 250,
            "legend_fontsize": "xx-large",
            "legend_loc": "upper right",
        },
        "graph": {
            "figsize": [20, 15],
            "n_neighbors": 5,
            "min_dist": 1,
            "n_components": 2,
            "random_state": 0,
            "scale_factor": 1.0,
            "node_size": 24,
            "font_size": 9,
            "shift_dir": "nw",
            "shift_len": 0.1,
            "edge_width": 0.5,
            "legend_marker_size": 120,
            "legend_fontsize": "x-large",
            "legend_loc": "upper right",
        },
    },
    "output": {
        "save_dir": "results",
        "formats": ["png", "svg"],
        "dpi": 150,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = base.copy()
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(config_path: Path | None) -> dict:
    """Load YAML config, falling back to built‑in defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        config = _deep_merge(config, user)
    return config


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _intersect_books(
    data_rm: np.lib.npyio.NpzFile,
    data_it: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    """Return index arrays so that books_rm[idxs_rm] == books_it[idxs_it]."""
    books_rm = data_rm["doc_names"]
    books_it = data_it["doc_names"]
    idxs_rm, idxs_it = [], []
    for i, b in enumerate(books_rm):
        j, = np.nonzero(books_it == b)
        if len(j) > 0:
            idxs_rm.append(i)
            idxs_it.append(j[0])
    return np.array(idxs_rm), np.array(idxs_it)


def _shorten_printer_names(
    names: np.ndarray,
    mode: str = "last_word",
    short_names: dict | None = None,
) -> np.ndarray:
    """Shorten printer names for display.

    If *short_names* is a dict, it is used as an explicit lookup table
    (full name → display name).  Any name not in the dict falls through
    to the *mode* rule.
    """
    if short_names is None:
        short_names = {}
    out = []
    for n in names:
        if n in short_names:
            out.append(short_names[n])
        elif mode == "last_word" and n:
            out.append(n.split()[-1])
        elif mode == "sf_ll":
            # First letter of first name, followed by last name
            parts = n.split()
            if len(parts) >= 2:
                out.append(f"{parts[0][0]}. {parts[-1]}")
            else:
                out.append(n)
        else:
            out.append(n)
    return np.array(out)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _save_figure(
    fig: matplotlib.figure.Figure,
    stem: str,
    save_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    """Save *fig* in every requested format."""
    for fmt in formats:
        out = save_dir / f"{stem}.{fmt}"
        fig.savefig(str(out), bbox_inches="tight", dpi=dpi)
        print(f"    Saved: {out}")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_corpus(
    corpus_dir: Path,
    config: dict,
    *,
    load_results: bool = False,
    plots: list[str] | None = None,
    report_dir: str | None = None,
) -> int:
    """Run the full a contrario analysis on a corpus directory.

    Parameters
    ----------
    corpus_dir : Path
        Must contain ``distances_roman.npz`` and/or ``distances_italic.npz``.
    config : dict
        Merged configuration dictionary.
    load_results : bool
        If *True*, load previously saved intermediate results instead of
        re‑computing them.
    plots : list[str] | None
        Which visualisations to produce (``"matrix"``, ``"graph"``).
        *None* means all.

    Returns
    -------
    0 on success, 1 on error.
    """
    if plots is None:
        plots = ["matrix", "graph"]

    # ---- Output directory ----
    save_dir = corpus_dir / config["output"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    formats = config["output"]["formats"]
    dpi = config["output"]["dpi"]

    # ---- Discover available styles ----
    path_rm = corpus_dir / "distances_roman.npz"
    path_it = corpus_dir / "distances_italic.npz"
    has_rm = path_rm.exists()
    has_it = path_it.exists()

    if not has_rm and not has_it:
        print("ERROR: No distance files found in", corpus_dir, file=sys.stderr)
        return 1

    results_path = save_dir / "acontrario_results.npz"

    # ---- Load data ----
    print("\nLoading distance matrices …")
    data_rm = np.load(str(path_rm), allow_pickle=True) if has_rm else None
    data_it = np.load(str(path_it), allow_pickle=True) if has_it else None

    # Common book list (intersection when both styles exist)
    if has_rm and has_it:
        idxs_rm, idxs_it = _intersect_books(data_rm, data_it)
        books = data_it["doc_names"][idxs_it]
        printers_raw = data_rm["printer_names"]
        remarks = data_rm["remarks"].copy()
        csv_indices = data_rm["indices"]
        # Tag known printers lacking a remark
        for i in range(len(remarks)):
            if printers_raw[i] != "unknown" and remarks[i] == "nan":
                remarks[i] = "known"
    elif has_rm:
        books = data_rm["doc_names"]
        printers_raw = data_rm["printer_names"]
        remarks = data_rm["remarks"].copy()
        csv_indices = data_rm["indices"]
        for i in range(len(remarks)):
            if printers_raw[i] != "unknown" and remarks[i] == "nan":
                remarks[i] = "known"
        idxs_rm = np.arange(len(books))
        idxs_it = None
    else:
        books = data_it["doc_names"]
        printers_raw = data_it["printer_names"]
        remarks = data_it["remarks"].copy()
        csv_indices = data_it["indices"]
        for i in range(len(remarks)):
            if printers_raw[i] != "unknown" and remarks[i] == "nan":
                remarks[i] = "known"
        idxs_rm = None
        idxs_it = np.arange(len(books))

    print(f"  Books: {len(books)}")
    
    # Change all "hidden" remarks to "known" since we don't distinguish them in the visualisation
    remarks = np.where(remarks == "hidden", "known", remarks)

    # ---- Shorten printer names for display ----
    short_mode = config["printers"].get("short_name", "last_word")
    short_names = config["printers"].get("short_names", None)
    printers = _shorten_printer_names(printers_raw, short_mode, short_names)
    printer_to_color = config["printers"].get("colors", {})
    remark_to_shape = config["remarks"]["shapes"]

    # When marker_style is "simple", override all shapes to circles
    marker_style = config["remarks"].get("marker_style", "by_remark")
    if marker_style == "simple":
        remark_to_shape = {k: "o" for k in remark_to_shape}
        remark_to_shape["nan"] = "o"

    # ---- Load adjacencies ----
    ds_rm = ds_it = None
    if has_rm:
        ds_rm = load_adjacencies(data_rm)
        mesh_rm = np.meshgrid(idxs_rm, idxs_rm)
        ds_rm = ds_rm[:, mesh_rm[0], mesh_rm[1]]
    if has_it:
        ds_it = load_adjacencies(data_it)
        mesh_it = np.meshgrid(idxs_it, idxs_it)
        ds_it = ds_it[:, mesh_it[0], mesh_it[1]]

    # ---- A contrario ----
    acfg = config["analysis"]
    alphas = build_alpha_grid(
        acfg["alphas"]["base_factors"],
        acfg["alphas"]["exponent_range"],
    )
    epsilon = acfg["epsilon"]
    N = len(books) * (len(books) - 1) // 2

    if load_results and results_path.exists():
        print("\nLoading cached results from", results_path, "…")
        cached = np.load(str(results_path), allow_pickle=True)
        n1hat_rm = cached.get("n1hat_roman")
        n_rm = cached.get("n_roman")
        n1hat_it = cached.get("n1hat_italic")
        n_it = cached.get("n_italic")
    else:
        # ---- Optional reference corpus for cross-corpus quantiles ----
        ref_path = acfg.get("reference_corpus")
        ref_ds_rm = ref_letters_rm = ref_ds_it = ref_letters_it = None
        if ref_path is not None:
            ref_dir = Path(ref_path).resolve()
            print(f"\n  Reference corpus: {ref_dir}")
            ref_rm_path = ref_dir / "distances_roman.npz"
            ref_it_path = ref_dir / "distances_italic.npz"
            if ref_rm_path.exists():
                ref_data_rm = np.load(str(ref_rm_path), allow_pickle=True)
                ref_ds_rm = load_adjacencies(ref_data_rm)
                ref_letters_rm = ref_data_rm["letters"]
            if ref_it_path.exists():
                ref_data_it = np.load(str(ref_it_path), allow_pickle=True)
                ref_ds_it = load_adjacencies(ref_data_it)
                ref_letters_it = ref_data_it["letters"]

        # Target corpus letters
        letters_rm = data_rm["letters"] if has_rm else None
        letters_it = data_it["letters"] if has_it else None

        n1hat_rm = n_rm = n1hat_it = n_it = None
        if ds_rm is not None:
            print("\nRunning a contrario — roman …")
            t0 = time.time()
            if ref_ds_rm is not None and letters_rm is not None:
                qs_rm, n_ref, n_self = estimate_quantiles_cross_corpus(
                    ds_rm, letters_rm, ref_ds_rm, ref_letters_rm, alphas,
                )
                print(f"  Quantiles: {n_ref} letters from reference, "
                      f"{n_self} from target")
            else:
                qs_rm = estimate_all_quantiles_(ds_rm, alphas=alphas)
            n_rm, n1hat_rm, _ = run_acontrario(
                ds_rm, qs_rm, alphas=alphas, N=N, epsilon=epsilon,
            )
            print(f"  Done in {time.time() - t0:.1f}s")
        if ds_it is not None:
            print("\nRunning a contrario — italic …")
            t0 = time.time()
            if ref_ds_it is not None and letters_it is not None:
                qs_it, n_ref, n_self = estimate_quantiles_cross_corpus(
                    ds_it, letters_it, ref_ds_it, ref_letters_it, alphas,
                )
                print(f"  Quantiles: {n_ref} letters from reference, "
                      f"{n_self} from target")
            else:
                qs_it = estimate_all_quantiles_(ds_it, alphas=alphas)
            n_it, n1hat_it, _ = run_acontrario(
                ds_it, qs_it, alphas=alphas, N=N, epsilon=epsilon,
            )
            print(f"  Done in {time.time() - t0:.1f}s")

        # ---- Save intermediate results ----
        save_kw: dict = {"books": books, "printers": printers, "remarks": remarks}
        if n1hat_rm is not None:
            save_kw.update(n1hat_roman=n1hat_rm, n_roman=n_rm)
        if n1hat_it is not None:
            save_kw.update(n1hat_italic=n1hat_it, n_italic=n_it)
        np.savez(str(results_path), **save_kw)
        print(f"\n  Saved intermediate results: {results_path}")

    # ---- Weights ----
    w_rm = (np.where(n1hat_rm > 0, n1hat_rm / n_rm, 0)
            if n1hat_rm is not None else None)
    w_it = (np.where(n1hat_it > 0, n1hat_it / n_it, 0)
            if n1hat_it is not None else None)

    # ---- Select weight matrix for ordering / graph distances ----
    ordering_mode = config["analysis"].get("ordering", "hierarchical")
    metric_mode = config["analysis"].get("metric", "average")  # "ordering" or "combined"
    if metric_mode == "roman":
        if w_rm is not None:
            w_order = w_rm
        else:
            print("  WARNING: ordering='roman' but no roman data; "
                  "falling back to italic")
            w_order = w_it
    elif metric_mode == "italic":
        if w_it is not None:
            w_order = w_it
        else:
            print("  WARNING: ordering='italic' but no italic data; "
                  "falling back to roman")
            w_order = w_rm
    else:  # "average" (default)
        if w_rm is not None and w_it is not None:
            w_order = (w_rm + w_it) / 2
        elif w_rm is not None:
            w_order = w_rm
        else:
            w_order = w_it
    print(f"  Metric mode: {metric_mode}")

    # Combined weight for filtering isolated books (uses all available data)
    if w_rm is not None and w_it is not None:
        w_combined = w_rm + w_it
    elif w_rm is not None:
        w_combined = w_rm
    else:
        w_combined = w_it

    # ---- Remove isolated books & compute ordering ----
    idxs_show, = np.nonzero(np.sum(w_combined > 0, axis=0) > 1)
    n_isolated = len(books) - len(idxs_show)
    if n_isolated:
        print(f"\n  {n_isolated} isolated book(s) removed")

    metric = 1 - w_order
    # Ensure metric is valid (cap at [0, 1])
    metric = np.clip(metric, 0, 1)
    if ordering_mode == "hierarchical":
        idxs_order = hierarchical_olo_order(metric)
        rel_order = np.array([np.argwhere(idxs_order == i)[0, 0] for i in range(len(books))])
    else:
        idxs_order = np.array(csv_indices, dtype=int) - 1  # use original CSV order (minus 1 for 0‑based)
        rel_order = idxs_order.copy()
        idxs_order = [np.argwhere(idxs_order == i)[0, 0] for i in range(len(books))]  # sort by CSV order
        
    idxs_order = np.array(idxs_order)
    rel_order = np.array(rel_order)

    # ---- Plotting ----
    vis = config["visualization"]
    mat_cfg = vis["matrix"]
    graph_cfg = vis["graph"]

    style_data = []
    if n1hat_rm is not None:
        style_data.append(("round", n1hat_rm, w_rm))
    if n1hat_it is not None:
        style_data.append(("cursive", n1hat_it, w_it))

    # ---- Matrix plots ----
    if "matrix" in plots:
        print("\nGenerating matrix plots …")
        mesh_order = np.meshgrid(idxs_order, idxs_order)
        figsize_side = mat_cfg["figsize_per_book"] * len(idxs_order)
        figsize = (figsize_side, figsize_side)
        for style_name, n1hat, _ in style_data:
            fig = plot_matrix(
                n1hat[mesh_order[0], mesh_order[1]],
                printers=printers[idxs_order],
                remarks=remarks[idxs_order],
                printer_to_color=printer_to_color,
                remark_to_shape=remark_to_shape,
                display_labels=None,
                include_values=mat_cfg["include_values"],
                title=f"$\\hat{{n}}_1$ — {style_name}",
                figsize=figsize,
                cmap=mat_cfg["cmap"],
                xticks_rotation=mat_cfg["xticks_rotation"],
                colorbar=mat_cfg["colorbar"],
                colorbar_step=mat_cfg["colorbar_step"],
                marker_edge_color=mat_cfg.get("marker_edge_color", "white"),
                marker_edge_width=mat_cfg.get("marker_edge_width", 0.4),
                marker_sizes=mat_cfg.get("marker_sizes", {}),
                legend_marker_size=mat_cfg.get("legend_marker_size", 250),
                legend_fontsize=mat_cfg.get("legend_fontsize", "xx-large"),
                legend_loc=mat_cfg.get("legend_loc", "upper right"),
            )
            _save_figure(fig, f"matrix_{style_name}", save_dir, formats, dpi)
            plt.close(fig)

    # ---- Graph plots ----
    if "graph" in plots:
        print("\nGenerating graph plots …")
        dists = metric.copy()
        np.fill_diagonal(dists, 0)
        graph_kw = {k: v for k, v in graph_cfg.items()}
        graph_kw["figsize"] = tuple(graph_kw["figsize"])

        for style_name, _, weights in style_data:
            fig = plot_graph(
                rel_order,
                dists,
                weights,
                printers=printers,
                remarks=remarks,
                printer_to_color=printer_to_color,
                remark_to_shape=remark_to_shape,
                idxs=idxs_order,
                **graph_kw,
            )
            _save_figure(fig, f"graph_{style_name}", save_dir, formats, dpi)
            plt.close(fig)

    print("\nDone.")

    # ---- Build step report -------------------------------------------------
    report = StepReport("acontrario")
    report_data: dict = {
        "n_documents": int(len(books)),
        "n_isolated_removed": int(n_isolated),
        "ordering_method": ordering_mode,
        "metric_mode": metric_mode,
        "epsilon": epsilon,
    }
    if n1hat_rm is not None:
        sig_rm = int((n1hat_rm < n_rm).sum()) // 2 if n_rm is not None else 0
        report_data["roman"] = {
            "n_significant_pairs": sig_rm,
            "mean_n1hat": round(float(n1hat_rm.mean()), 4),
        }
    if n1hat_it is not None:
        sig_it = int((n1hat_it < n_it).sum()) // 2 if n_it is not None else 0
        report_data["italic"] = {
            "n_significant_pairs": sig_it,
            "mean_n1hat": round(float(n1hat_it.mean()), 4),
        }
    # List generated plots
    generated_plots = [str(p) for p in save_dir.glob("*.png")] + \
                      [str(p) for p in save_dir.glob("*.svg")]
    report_data["output_plots"] = generated_plots
    report.set_summary(report_data)

    report_dir = Path(report_dir) if report_dir else corpus_dir / "reports"
    rpath = report.save(report_dir)
    print(f"Report saved: {rpath}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="A contrario analysis and visualisation of typographic distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_acontrario.py data/corpus-1 --config configs/acontrario_corpus1.yaml
  python scripts/run_acontrario.py data/corpus-2 --config configs/acontrario_corpus2.yaml
  python scripts/run_acontrario.py data/corpus-1 --config configs/acontrario_corpus1.yaml --plots matrix
  python scripts/run_acontrario.py data/corpus-1 --config configs/acontrario_corpus1.yaml --load-results
        """,
    )
    parser.add_argument(
        "corpus_dir",
        help="Corpus directory (must contain distances_roman.npz and/or distances_italic.npz)",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/acontrario_corpus1.yaml)",
    )
    parser.add_argument(
        "--plots",
        default="matrix,graph",
        help="Comma‑separated list of plots to produce: matrix, graph (default: matrix,graph)",
    )
    parser.add_argument(
        "--load-results",
        action="store_true",
        help="Load previously saved intermediate results instead of re‑computing",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Use non‑interactive Matplotlib backend (for headless servers)",
    )
    parser.add_argument(
        "--report-dir",
        help="Directory for the step report JSON (default: {corpus}/reports)",
    )

    args = parser.parse_args(argv)

    if args.no_display:
        matplotlib.use("Agg")

    corpus_dir = Path(args.corpus_dir).resolve()
    if not corpus_dir.is_dir():
        print(f"ERROR: directory not found: {corpus_dir}", file=sys.stderr)
        return 1

    config = load_config(Path(args.config))
    plots = [p.strip() for p in args.plots.split(",")]

    return process_corpus(
        corpus_dir, config, load_results=args.load_results, plots=plots,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
