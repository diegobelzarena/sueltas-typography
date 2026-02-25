# Sueltas Typography

> **Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography**
>
> Official implementation — ICDAR 2026

Unsupervised pipeline for extracting, clustering, and comparing typographic features from historical printed documents.

---

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU recommended.

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

Run the complete pipeline on a corpus with a single command:

```bash
python scripts/run_pipeline.py data/corpus-1 --steps 1,2,3,4 --workers 4
```

Or run individual steps (see below for details).

---

## Pipeline Overview

The pipeline assumes document images (PNG) are already available, organized as:
```
data/corpus-1/imgs/
    document_001/
        page_001.png
        page_002.png
        ...
    document_002/
        ...
```

### Step 1 — OCR with CharNet

Detect characters and words using the CharNet neural network.

```bash
python scripts/run_charnet.py configs/icdar2015_hourglass88.yaml \
    data/corpus-1/imgs  data/corpus-1/charnet \
    --workers 4
```

**Output:** `data/corpus-1/charnet/{document}/{page}.json` — word/character bounding boxes with recognition scores.

---

### Step 2 — Character Extraction via Minimum Cost Paths

Segment individual characters using graph-based minimum cost path algorithm, compute page orientations via FFT, and embed characters to fixed-size images.

```bash
python scripts/character_extraction.py \
    data/corpus-1/imgs  data/corpus-1/charnet \
    --workers 4
```

**Output:** `{page}_data.npz` files containing:
- `char_imgs` — embedded character images (40×32, normalized)
- `char_labels` — OCR labels
- `word_orientations` — page orientation per word (FFT-based)
- `word_stroke_orientations` — stroke angle per word (structure tensor)

---

### Step 3 — Italic Detection via Structure Tensor

Classify characters as italic or round based on stroke orientation distribution.

```bash
python scripts/italic_detection.py data/corpus-1/charnet \
    --process-subfolders
```

**Output:** `italic_labels.npz` per document containing:
- `char_italic` — boolean array (True = italic)
- `threshold` — optimal separation angle

---

### Step 4 — Unsupervised Tree Clustering

Cluster character images using GMM initialization and tree-based refinement.

```bash
python scripts/clustering.py data/corpus-1/charnet \
    --process-subfolders --workers 8
```

**Output:** `clusters_all.npz` per document containing:
- `cluster_means` — cluster centroids (40×32 images)
- `cluster_labels` — per-cluster `[label, confidence, count]`
- `cluster_italic` — mean italic ratio per cluster

---

### Step 5 — Typographic Distance Computation

Compare documents by computing pairwise distances between cluster centroids.

```bash
python scripts/typographic_distances.py data/corpus-1
```

This processes both roman and italic styles by default. To process a single style:

```bash
python scripts/typographic_distances.py data/corpus-1 --style roman
```

**Configuration:** Parameters can be customized via `configs/typographic_distances.yaml`.

**Output:** `distances_roman.npz` and `distances_italic.npz` at corpus root containing:
- `doc_names` — document identifiers
- `printer_names` — printer attribution from metadata CSV
- `adjacencies` — (n_letters, n_docs, n_docs) distance matrices per letter
- `letters` — characters used in comparison

---

## Notebooks

Interactive notebooks for visualization and debugging:

| Notebook | Description |
|----------|-------------|
| `visualize_char_extraction.ipynb` | Inspect extracted characters, bounding boxes, and orientation data |
| `visualize_italic.ipynb` | Visualize italic/round classification with stroke histograms |
| `visualize_clustering.ipynb` | Browse cluster means, character assignments, and compare documents |
| `visualize_detections.ipynb` | Browse CharNet OCR detections overlaid on page images |
| `debug_char_segment.ipynb` | Debug character segmentation algorithm |
| `debug_typographic_distances.ipynb` | Debug distance computation and per-letter analysis |

---

## Utilities

### PDF to PNG Conversion

Convert scanned PDFs to normalized PNG page images at a target DPI:

```bash
# Basic conversion
python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs

# With precomputed DPI values and parallel processing
python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs \
    --dpi-csv-dir data/corpus-1/dpis --workers 8 --skip-existing

# Custom target DPI
python scripts/convert_pdfs.py data/corpus-1/pdfs data/corpus-1/imgs --target-dpi 300
```

### DPI Estimation

Compute original DPI and physical page dimensions for PDF or TIFF collections:

```bash
# Single CSV report
python scripts/compute_dpi.py data/corpus-1/pdfs -o dpi_report.csv

# Per-PDF CSV files (for use with convert_pdfs.py --dpi-csv-dir)
python scripts/compute_dpi.py data/corpus-1/pdfs -o data/corpus-1/dpis --per-file --skip-existing
```

### Catalogue Search

Find PDFs matching catalogue signatures and optionally copy them:

```bash
python scripts/search_pdfs.py catalogue.csv data/pdfs/ --out data/selected/ --skip-existing
```

### Validate Pipeline Outputs

```bash
python scripts/validate_outputs.py data/corpus-1/charnet/doc001
python scripts/validate_outputs.py data/corpus-1/charnet --all
```

---

## Repository Structure

```
scripts/           # Pipeline entry points (self-contained)
configs/           # YAML configuration files
src/
  ├── charnet_src/ # CharNet OCR module (third-party)
  └── image_processing/  # Orientation, segmentation, clustering
notebooks/         # Visualization notebooks
data/              # Input/output data (not tracked)
```

---

## Citation

```bibtex
@inproceedings{sueltas2026,
  title={Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography},
  author={...},
  booktitle={ICDAR},
  year={2026}
}
```
