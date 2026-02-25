#!/usr/bin/env python
"""Generate illustrative figures for the README from example data.

Produces one PNG per pipeline step, saved to docs/images/.

Usage
-----
    python scripts/generate_readme_images.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DOC = ROOT / "examples" / "charnet" / "BNE_1001_615_T-55281-18"
EXAMPLE_IMG = ROOT / "examples" / "imgs" / "BNE_1001_615_T-55281-18"
CORPUS1     = ROOT / "data" / "corpus-1"
OUT_DIR     = ROOT / "docs" / "images"

DPI = 150


def save(fig, name):
    out = OUT_DIR / f"{name}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.relative_to(ROOT)}")


# ===================================================================
# Step 1 — CharNet OCR detections
# ===================================================================
def generate_step1():
    print("Step 1: CharNet detections …")
    img_path = EXAMPLE_IMG / "page_10.png"
    json_path = EXAMPLE_DOC / "page_10.json"
    if not img_path.exists() or not json_path.exists():
        print("  SKIP — example files not found")
        return

    img = plt.imread(str(img_path))
    with open(json_path) as f:
        detections = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(img, cmap="gray")

    # Draw a subset of word bounding boxes
    for det in detections[:80]:
        t, b, l, r = det["tblr"]
        rect = patches.Rectangle((l, t), r - l, b - t,
                                  linewidth=0.8, edgecolor="tab:red",
                                  facecolor="none", alpha=0.7)
        ax.add_patch(rect)
        ax.text(l, t - 2, det["text"], fontsize=3.5, color="tab:blue",
                va="bottom", clip_on=True)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    ax.set_title("Step 1 — CharNet word detections", fontsize=12)
    save(fig, "step1_charnet")


# ===================================================================
# Step 2 — Extracted characters
# ===================================================================
def generate_step2():
    print("Step 2: Extracted characters …")
    npz_path = EXAMPLE_DOC / "page_10_data.npz"
    if not npz_path.exists():
        print("  SKIP — data not found")
        return

    data = np.load(str(npz_path), allow_pickle=True)
    imgs = data["char_imgs"]
    labels = data["char_labels"]

    # Pick a diverse sample: first 60 characters
    n_show = min(60, len(imgs))
    ncols = 15
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 0.8))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_show:
            ax.imshow(imgs[i], cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(labels[i], fontsize=7, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Step 2 — Extracted character images (40×32)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "step2_characters")


# ===================================================================
# Step 3 — Italic detection
# ===================================================================
def generate_step3():
    print("Step 3: Italic detection …")
    italic_path = EXAMPLE_DOC / "italic_labels.npz"
    page_path = EXAMPLE_DOC / "page_10_data.npz"
    if not italic_path.exists() or not page_path.exists():
        print("  SKIP — data not found")
        return

    ital = np.load(str(italic_path), allow_pickle=True)
    page_data = np.load(str(page_path), allow_pickle=True)

    # Compute offset for page_10 in the document-level italic array
    page_names = list(ital["page_names"])
    page_counts = ital["page_char_counts"]
    if "page_10" in page_names:
        page_idx = page_names.index("page_10")
        offset = int(page_counts[:page_idx].sum())
        count = int(page_counts[page_idx])
    else:
        offset, count = 0, min(60, len(ital["char_italic"]))

    italic_labels = ital["char_italic"][offset:offset + count]
    imgs = page_data["char_imgs"][:count]
    labels = page_data["char_labels"][:count]

    # Show 30 roman, 30 italic (or as many as available)
    np.random.seed(0)
    roman_idx = np.where(italic_labels == 0)[0][np.random.choice(np.where(italic_labels == 0)[0].shape[0], min(30, np.sum(italic_labels == 0)), replace=False)]
    italic_idx = np.where(italic_labels == 1)[0][np.random.choice(np.where(italic_labels == 1)[0].shape[0], min(30, np.sum(italic_labels == 1)), replace=False)]
    
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 3))

    def _draw_row(ax, indices, title, border_color):
        n = len(indices)
        ax.set_xlim(0, max(n, 1))
        ax.set_ylim(0, 1)
        for k, idx in enumerate(indices):
            if idx < len(imgs):
                inset = ax.inset_axes([k / max(n, 1), 0, 1 / max(n, 1), 1])
                inset.imshow(imgs[idx], cmap="gray_r", vmin=0, vmax=1)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(1.5)
        ax.set_title(title, fontsize=10, color=border_color)
        ax.axis("off")

    _draw_row(ax_top, roman_idx, "Roman characters", "tab:blue")
    _draw_row(ax_bot, italic_idx, "Italic characters", "tab:red")

    fig.suptitle("Step 3 — Italic / Roman classification", fontsize=12, y=1.04)
    fig.tight_layout()
    save(fig, "step3_italic")


# ===================================================================
# Step 4 — Cluster means
# ===================================================================
def generate_step4():
    print("Step 4: Cluster means …")
    cluster_path = EXAMPLE_DOC / "clusters_all.npz"
    if not cluster_path.exists():
        print("  SKIP — data not found")
        return

    data = np.load(str(cluster_path), allow_pickle=True)
    means = data["cluster_means"]
    labels = data["cluster_labels"]

    # Show top 60 clusters sorted by count (column 2)
    counts = labels[:, 2].astype(float)
    order = np.argsort(counts)[::-1]
    n_show = min(60, len(means))
    ncols = 15
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 0.8))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_show:
            idx = order[i]
            ax.imshow(means[idx], cmap="gray_r")
            ax.set_title(f"{labels[idx, 0]}", fontsize=7, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Step 4 — Cluster mean images (top 60 by count)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "step4_clusters")


# ===================================================================
# Step 5 — Typographic distance matrix
# ===================================================================
def generate_step5():
    print("Step 5: Distance matrix …")
    dist_path = CORPUS1 / "distances_roman.npz"
    if not dist_path.exists():
        print("  SKIP — distances not computed")
        return

    data = np.load(str(dist_path), allow_pickle=True)
    adjs = data["adjacencies"]
    letters = data["letters"]

    # Show distance matrix for one common character (e.g. 'e' or first available)
    target = "e"
    if target in letters:
        idx = np.where(letters == target)[0][0]
    else:
        idx = 0
        target = letters[0]

    mat = adjs[idx]
    # Replace sentinel max values with NaN for display
    finite = mat[np.isfinite(mat)]
    if len(finite) > 0:
        vmax = np.percentile(finite[finite > 0], 95) if np.any(finite > 0) else 1
    else:
        vmax = 1
    display = np.where(np.isinf(mat) | (mat == mat.max()), np.nan, mat)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_title(f"Step 5 — Inter-document distances for letter '{target}'",
                 fontsize=11)
    ax.set_xlabel("Document index")
    ax.set_ylabel("Document index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine distance")
    save(fig, "step5_distances")


# ===================================================================
# Step 6 — A contrario results
# ===================================================================
def generate_step6():
    print("Step 6: A contrario results …")
    results_dir = CORPUS1 / "results"
    matrix_path = results_dir / "matrix_round.png"
    graph_path = results_dir / "graph_round.png"

    if not matrix_path.exists() and not graph_path.exists():
        print("  SKIP — results not generated")
        return

    images = []
    titles = []
    if matrix_path.exists():
        images.append(plt.imread(str(matrix_path)))
        titles.append("$\\hat{n}_1$ matrix (round)")
    if graph_path.exists():
        images.append(plt.imread(str(graph_path)))
        titles.append("Document similarity graph (round)")

    fig, axes = plt.subplots(1, len(images), figsize=(7 * len(images), 7))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Step 6 — A contrario analysis", fontsize=13, y=1.0)
    fig.tight_layout()
    save(fig, "step6_acontrario")


# ===================================================================
# Main
# ===================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_DIR.relative_to(ROOT)}/\n")

    generate_step1()
    generate_step2()
    generate_step3()
    generate_step4()
    generate_step5()
    generate_step6()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
