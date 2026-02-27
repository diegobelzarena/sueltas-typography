#!/usr/bin/env python
"""Generate illustrative figures for the README from example data.

Produces one PNG per pipeline step, saved to docs/images/.

Usage
-----
    python scripts/generate_readme_images.py
"""

from __future__ import annotations

import csv
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

# ---------------------------------------------------------------------------
# Step 4 configuration — italic cluster specimen across documents
# ---------------------------------------------------------------------------
# Letters to show (rows) and their order.
STEP4_LETTERS = ["a", "d", "e", "o", "r"]

# Corpus whose charnet/ folder contains the per-document clusters_all.npz.
# Change to "corpus-2" when data is available.
STEP4_CORPUS = ROOT / "data" / "corpus-2"

# Metadata CSV with columns Index, FileName, Printer, …
STEP4_CSV = STEP4_CORPUS / "ordered-table-corpus2.csv"

# Ordered list of document folder names (columns).
# Set to None to use every document that has cluster data, sorted by CSV index.
STEP4_DOC_ORDER: list[str] | None = None

# Maximum number of document columns to show (when STEP4_DOC_ORDER is None).
STEP4_MAX_DOCS = 20


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

def _step2_common():
    """Load data shared by all step-2 variants."""
    npz_path = EXAMPLE_DOC / "page_10_data.npz"
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path), allow_pickle=True)
    return data["char_imgs"], data["char_labels"]


def generate_step2_grouped():
    """Variant A: characters grouped by OCR label in a type-case layout."""
    print("Step 2A: Grouped by letter …")
    result = _step2_common()
    if result is None:
        print("  SKIP"); return
    imgs, labels = result

    # Pick the 8 most frequent letters
    unique, counts = np.unique(labels, return_counts=True)
    # Filter to alphabetic chars only
    mask = np.array([u.isalpha() for u in unique])
    unique, counts = unique[mask], counts[mask]
    top_letters = unique[np.argsort(counts)[::-1]][:8]

    n_per_letter = 6
    nrows = len(top_letters)

    fig, axes = plt.subplots(nrows, n_per_letter + 1, figsize=(8, nrows * 1.0),
                             gridspec_kw={"width_ratios": [1.2] + [1]*n_per_letter,
                                          "wspace": 0.08, "hspace": 0.25})
    for r, letter in enumerate(top_letters):
        # Label column
        axes[r, 0].text(0.5, 0.5, letter, fontsize=18, ha="center", va="center",
                        fontfamily="serif", fontweight="bold", color="#333")
        axes[r, 0].axis("off")
        # Character samples
        idxs = np.where(labels == letter)[0]
        np.random.seed(42 + r)
        chosen = np.random.choice(idxs, min(n_per_letter, len(idxs)), replace=False)
        for c in range(n_per_letter):
            ax = axes[r, c + 1]
            if c < len(chosen):
                ax.imshow(imgs[chosen[c]], cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Extracted character images (40×32)", fontsize=13, y=0.98,
                 fontweight="bold", color="#222")
    save(fig, "step2_grouped")


def generate_step2_beforeafter():
    """Variant B: raw crops → normalised images transformation."""
    print("Step 2B: Before/after …")
    npz_path = EXAMPLE_DOC / "page_10_data.npz"
    img_path = EXAMPLE_IMG / "page_10.png"
    json_path = EXAMPLE_DOC / "page_10.json"
    if not all(p.exists() for p in [npz_path, img_path, json_path]):
        print("  SKIP"); return

    page_img = plt.imread(str(img_path))
    # if page_img.ndim == 3:
    #     page_gray = np.mean(page_img[..., :3], axis=-1)
    # else:
    #     page_gray = page_img
    data = np.load(str(npz_path), allow_pickle=True)
    imgs = data["char_imgs"]
    labels = data["char_labels"]

    # Choose 15 diverse characters (different labels)
    n_show = 15
    seen = set()
    chosen = []
    for i, lab in enumerate(labels):
        if i < 100:
            continue  # skip first few chars which are often partial/non-alpha
        if lab.isalpha() and lab not in seen:
            # Get raw crop from bounding box
            word_idx = data["char_word_idx"][i]
            key = f"char_tblrs_{word_idx}"
            if key in data:
                tblrs = data[key]
                mask_key = f"mask_idx_{word_idx}"
                if mask_key in data:
                    mask_idx = data[mask_key]
                    local = np.where(mask_idx == i)[0]
                    if len(local) > 0:
                        t, b, l, r = tblrs[local[0]]
                        if b > t and r > l and b < page_img.shape[0] and r < page_img.shape[1]:
                            chosen.append((i, t, b, l, r))
                            seen.add(lab)
            if len(chosen) >= n_show:
                break

    if len(chosen) < 4:
        print("  SKIP — not enough crops found"); return

    nshow = len(chosen)
    fig, axes = plt.subplots(2, nshow, figsize=(nshow * 0.9, 2.4),
                             gridspec_kw={"hspace": 0.15, "wspace": 0.08})

    for c, (idx, t, b, l, r) in enumerate(chosen):
        # Raw crop
        crop = page_img[t:b, l:r]
        axes[0, c].imshow(crop, cmap="gray", vmin=0, vmax=crop.max() if crop.max() > 0 else 1)
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        for sp in axes[0, c].spines.values():
            sp.set_color("#bbb"); sp.set_linewidth(0.5)

        # Normalised
        axes[1, c].imshow(imgs[idx], cmap="gray_r")
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
        axes[1, c].set_xlabel(labels[idx], fontsize=9, fontfamily="serif")
        for sp in axes[1, c].spines.values():
            sp.set_color("#bbb"); sp.set_linewidth(0.5)

    axes[0, 0].set_ylabel("raw crop", fontsize=9, color="#666")
    axes[1, 0].set_ylabel("40×32", fontsize=9, color="#666")
    fig.suptitle("Character extraction: raw → normalised",
                 fontsize=13, fontweight="bold", color="#222", y=1.02)
    save(fig, "step2_beforeafter")


def generate_step2_grid():
    """Variant C: clean flat grid with fewer chars, better spacing."""
    print("Step 2C: Clean grid …")
    result = _step2_common()
    if result is None:
        print("  SKIP"); return
    imgs, labels = result

    # Select 40 characters, preferring variety of labels
    np.random.seed(42)
    unique_labels = np.unique(labels)
    alpha = [l for l in unique_labels if l.isalpha()]
    chosen = []
    for lab in alpha:
        idxs = np.where(labels == lab)[0]
        chosen.append(np.random.choice(idxs))
        if len(chosen) >= 40:
            break
    # Fill remainder randomly
    while len(chosen) < 40:
        i = np.random.randint(len(imgs))
        if i not in chosen:
            chosen.append(i)
    chosen = chosen[:40]

    ncols, nrows = 10, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 3.6),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.35})
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < len(chosen):
            ax.imshow(imgs[chosen[i]], cmap="gray_r")
            ax.set_title(labels[chosen[i]], fontsize=8, pad=2,
                         fontfamily="serif", color="#444")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle("Extracted character images (40×32)",
                 fontsize=13, fontweight="bold", color="#222", y=0.98)
    save(fig, "step2_grid")


def generate_step2():
    generate_step2_grouped()
    generate_step2_beforeafter()
    generate_step2_grid()


# ===================================================================
# Step 3 — Italic detection (histogram + examples)
# ===================================================================
def generate_step3():
    print("Step 3: Italic histogram + examples …")
    italic_path = EXAMPLE_DOC / "italic_labels.npz"
    if not italic_path.exists():
        print("  SKIP — data not found"); return

    ital = np.load(str(italic_path), allow_pickle=True)
    threshold = float(ital["threshold"][0])

    # Collect stroke orientations from all pages
    all_strokes = []
    all_italic_per_word = []
    page_names = list(ital["page_names"])
    page_counts = ital["page_char_counts"]

    # Also collect some example chars for display
    roman_examples = []
    italic_examples = []
    char_offset = 0

    for pname, pcount in zip(page_names, page_counts):
        pdata_path = EXAMPLE_DOC / f"{pname}_data.npz"
        if not pdata_path.exists():
            char_offset += int(pcount)
            continue
        pdata = np.load(str(pdata_path), allow_pickle=True)
        strokes = pdata["word_stroke_orientations"]
        char_word_idx = pdata["char_word_idx"]
        char_imgs = pdata["char_imgs"]
        char_italic = ital["char_italic"][char_offset:char_offset + int(pcount)]

        # filter valid strokes
        valid = (strokes != -360) & np.isfinite(strokes)
        all_strokes.append(strokes[valid])

        # Collect example characters
        for ci in range(min(int(pcount), len(char_imgs))):
            if ci >= len(char_italic):
                break
            if char_italic[ci] == 0 and len(roman_examples) < 15:
                lab = pdata["char_labels"][ci] if ci < len(pdata["char_labels"]) else ""
                if lab.isalpha():
                    roman_examples.append(char_imgs[ci])
            elif char_italic[ci] == 1 and len(italic_examples) < 15:
                lab = pdata["char_labels"][ci] if ci < len(pdata["char_labels"]) else ""
                if lab.isalpha():
                    italic_examples.append(char_imgs[ci])

        char_offset += int(pcount)

    all_strokes = np.concatenate(all_strokes) if all_strokes else np.array([])

    if len(all_strokes) == 0:
        print("  SKIP — no stroke data"); return

    # --- Figure: histogram on left, example chars on right ---
    fig = plt.figure(figsize=(11, 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], wspace=0.3, hspace=0.35)

    # Histogram
    ax_hist = fig.add_subplot(gs[:, 0])
    bins = np.linspace(-50, 50, 80)
    roman_mask = all_strokes < threshold
    italic_mask = all_strokes >= threshold

    ax_hist.hist(all_strokes[roman_mask], bins=bins, color="#4A90D9", alpha=0.7,
                 label="Roman", edgecolor="white", linewidth=0.3)
    ax_hist.hist(all_strokes[italic_mask], bins=bins, color="#E74C3C", alpha=0.7,
                 label="Italic", edgecolor="white", linewidth=0.3)
    ax_hist.axvline(threshold, color="#E74C3C", ls="--", lw=1.5, alpha=0.8)
    ax_hist.text(threshold + 0.8, ax_hist.get_ylim()[1] * 0.92,
                 f"θ = {threshold:.1f}°", fontsize=9, color="#E74C3C", va="top")

    ax_hist.set_xlabel("Stroke orientation (degrees)", fontsize=10)
    ax_hist.set_ylabel("Word count", fontsize=10)
    ax_hist.legend(fontsize=9, framealpha=0.9)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.set_title("Stroke angle distribution", fontsize=11,
                      fontweight="bold", color="#222")

    # Roman examples
    ax_rom = fig.add_subplot(gs[0, 1])
    _draw_char_strip(ax_rom, roman_examples[:12], "Roman samples", "#4A90D9")

    # Italic examples
    ax_ita = fig.add_subplot(gs[1, 1])
    _draw_char_strip(ax_ita, italic_examples[:12], "Italic samples", "#E74C3C")

    save(fig, "step3_italic")


def _draw_char_strip(ax, char_list, title, color):
    """Draw a horizontal strip of character images inside an axes."""
    n = len(char_list)
    if n == 0:
        ax.axis("off"); return

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10, color=color, fontweight="bold", loc="left")
    ax.axis("off")

    for k, img in enumerate(char_list):
        inset = ax.inset_axes([k / n, 0, 0.95 / n, 1.0])
        inset.imshow(img, cmap="gray_r")
        inset.set_xticks([]); inset.set_yticks([])
        for sp in inset.spines.values():
            sp.set_color(color); sp.set_linewidth(1.2)


# ===================================================================
# Step 4 — Italic cluster specimen across documents
# ===================================================================

def _load_doc_index_map(csv_path: Path) -> dict[str, int]:
    """Return {FileName: Index} from the ordered-table CSV."""
    mapping: dict[str, int] = {}
    if not csv_path.exists():
        return mapping
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row.get("FileName", "")
            try:
                mapping[fname] = int(row["Index"])
            except (KeyError, ValueError):
                pass
    return mapping


def _pick_italic_cluster(means, labels_arr, italics, letter: str):
    """Return the cluster-mean image for the highest-count italic cluster
    matching *letter*, or None if nothing matches."""
    char_labels = labels_arr[:, 0]
    counts = labels_arr[:, 2].astype(float)
    mask = (char_labels == letter) & (italics > 0.5)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    best = idxs[np.argmax(counts[idxs])]
    return means[best]


def generate_step4():
    """Italic-specimen grid: rows = letters, columns = documents."""
    print("Step 4: Italic specimen grid …")
    charnet_dir = STEP4_CORPUS / "charnet"
    if not charnet_dir.exists():
        print(f"  SKIP — {charnet_dir} does not exist"); return

    # --- Resolve document order ------------------------------------------------
    idx_map = _load_doc_index_map(STEP4_CSV)

    if STEP4_DOC_ORDER is not None:
        doc_dirs = [charnet_dir / d for d in STEP4_DOC_ORDER]
    else:
        # All docs with cluster data, sorted by CSV index (then by name)
        doc_dirs = sorted(
            [d for d in charnet_dir.iterdir()
             if d.is_dir() and (d / "clusters_all.npz").exists()],
            key=lambda d: (idx_map.get(d.name, 9999), d.name),
        )
        if STEP4_MAX_DOCS and len(doc_dirs) > STEP4_MAX_DOCS:
            doc_dirs = doc_dirs[:STEP4_MAX_DOCS]

    if not doc_dirs:
        print("  SKIP — no documents found"); return

    letters = STEP4_LETTERS
    ndocs = len(doc_dirs)
    nletters = len(letters)

    # --- Build cell images -----------------------------------------------------
    grid: list[list[np.ndarray | None]] = []  # [letter_row][doc_col]
    for letter in letters:
        row = []
        for d in doc_dirs:
            cp = d / "clusters_all.npz"
            if not cp.exists():
                row.append(None); continue
            data = np.load(str(cp), allow_pickle=True)
            img = _pick_italic_cluster(
                data["cluster_means"], data["cluster_labels"],
                data["cluster_italic"], letter,
            )
            row.append(img)
        grid.append(row)

    # --- Determine document labels (CSV index) ---------------------------------
    doc_labels = []
    for d in doc_dirs:
        idx = idx_map.get(d.name)
        doc_labels.append(str(idx) if idx is not None else d.name[:8])

    # --- Draw ------------------------------------------------------------------
    cell_px = 0.55  # inches per cell
    label_col_w = 0.7
    header_row_h = 0.35

    fig_w = label_col_w + ndocs * cell_px + 0.3
    fig_h = header_row_h + nletters * cell_px + 0.2

    fig, axes = plt.subplots(
        nletters, ndocs + 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "width_ratios": [label_col_w / cell_px] + [1] * ndocs,
            "wspace": 0.04,
            "hspace": 0.04,
        },
    )
    axes = np.atleast_2d(axes)

    for r, letter in enumerate(letters):
        # Letter label column
        axes[r, 0].text(
            0.6, 0.5, letter,
            fontsize=16, ha="center", va="center",
            fontfamily="serif", fontstyle="italic",
            fontweight="bold", color="#333",
        )
        axes[r, 0].axis("off")

        for c in range(ndocs):
            ax = axes[r, c + 1]
            img = grid[r][c]
            if img is not None:
                ax.imshow(img, cmap="gray_r")
            else:
                ax.set_facecolor("#f5f5f5")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            # Document index header (first row only)
            if r == 0:
                ax.set_title(
                    doc_labels[c], fontsize=6, pad=2,
                    color="#888", fontfamily="monospace",
                )

    fig.suptitle(
        "Italic cluster centroids across documents",
        fontsize=12, fontweight="bold", color="#222", y=0.99,
    )
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
