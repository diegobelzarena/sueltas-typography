# Sueltas Typography

> **Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography**
>
> Official implementation — ICDAR 2026

Unsupervised pipeline for extracting, clustering, and comparing typographic features from historical printed documents.

---

## Installation

**Requirements:** Python 3.11+, CUDA-capable GPU recommended.

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
- `cluster_labels` — cluster assignment per character
- `cluster_means` — cluster centroids (40×32 images)
- `cluster_counts` — characters per cluster

---

### Step 5 — Typographic Distance Computation

*Coming soon.*

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

---

## Utilities

### PDF to PNG Conversion

```bash
python scripts/convert_pdfs.py input_pdfs/ output_imgs/ --dpi-csv-dir dpis/
```

### DPI Estimation

```bash
python scripts/compute_dpi.py input_pdfs/ -o dpis/ --per-file
```

### Catalogue Search

```bash
python scripts/search_pdfs.py catalogue.csv pdf_folder/ --out selected/
```

### Validate Pipeline Outputs

```bash
python scripts/validate_outputs.py data/corpus-1/charnet/doc001
python scripts/validate_outputs.py data/corpus-1/charnet --all
```

---

## Repository Structure

```
scripts/           # Pipeline entry points
src/
  ├── charnet/     # CharNet OCR module
  ├── image_processing/  # Orientation, segmentation, clustering
  └── io/          # PDF/image utilities
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
