# Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography

> Official implementation — ICDAR 2026

Unsupervised pipeline for extracting, clustering, and comparing typographic features from historical printed documents.
## Dataset
The full dataset used in this project is available at:
[https://osf.io/tkwf9/overview?view_only=6fcc7fdf07ee444d86fddd1aefe11659](https://osf.io/tkwf9/overview?view_only=6fcc7fdf07ee444d86fddd1aefe11659)

<p align="center">
  <img src="docs/images/step6_acontrario.png" width="90%" alt="A contrario analysis results"/>
  <img src="docs/images/step4_clusters.png" width="90%" alt="A contrario analysis results"/>
</p>

---

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU recommended (CPU fallback supported but slow for CharNet).

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Configuration

All configuration files live in the `configs/` directory:

| File | Description |
|------|-------------|
| `configs/icdar2015_hourglass88.yaml` | CharNet model config (input size, model weights path, char dictionary, lexicon, detection thresholds) |
| `configs/pipeline_corpus2.yaml` | Pipeline config for DocTR detector + logit segmentation (corpus 2) |
| `configs/typographic_distances.yaml` | Typographic distance parameters (italic thresholds, filtering, registration, distance metric) |
| `configs/acontrario_corpus1.yaml` | A contrario analysis config for corpus 1 (analysis parameters, printer colours, figure settings) |
| `configs/acontrario_corpus2.yaml` | A contrario analysis config for corpus 2 |

**CharNet config** (`configs/icdar2015_hourglass88.yaml`): Paths for `WEIGHT`, `CHAR_DICT_FILE`, and `WORD_LEXICON_PATH` are relative to `src/charnet_src/` and are resolved automatically at runtime — the repository works from any location.

### Pipeline Config (CharNet vs DocTR)

By default the pipeline uses **CharNet** for OCR detection (Step 1) and
**box_init** segmentation (Step 2).  To use **DocTR** instead, pass a
pipeline config with `--pipeline-config`:

```yaml
# configs/pipeline_corpus2.yaml
detector: doctr            # "charnet" or "doctr"
segmentation: logit_init   # "box_init" or "logit_init"

doctr:
  det_arch: fast_base      # DocTR detection architecture

source_conversion:
  target_dpi: 150          # for Step 0 (--pdf-dir)
  dpi_csv_dir: null        # path to pre-computed DPI CSVs, or null
```

| Key | Values | Default | Effect |
|-----|--------|---------|--------|
| `detector` | `charnet`, `doctr` | `charnet` | Which OCR engine to use in Step 1 |
| `segmentation` | `box_init`, `logit_init` | `box_init` | Character segmentation method in Step 2 |
| `doctr.det_arch` | any [DocTR arch](https://mindee.github.io/doctr/) | `fast_base` | Detection backbone (only when `detector: doctr`) |
| `source_conversion.target_dpi` | integer | `150` | Target DPI for PDF/TIFF conversion (Step 0) |
| `source_conversion.dpi_csv_dir` | path or `null` | `null` | Pre-computed DPI CSVs directory |

You can also override individual config files via CLI flags (`--charnet-config`, `--acontrario-config`).

---

## Quick Start

Run the complete pipeline on a corpus with a single command:

```bash
# CharNet (default) — corpus with images already extracted
python scripts/run_pipeline.py data/corpus-1 --steps 1,2,3,4,5,6 --workers 4

# DocTR — use a pipeline config to switch detector and segmentation
python scripts/run_pipeline.py data/corpus-2 \
    --pipeline-config configs/pipeline_corpus2.yaml \
    --steps 1,2,3,4,5,6 --workers 4
```

### Starting from PDFs

If you have a folder of scanned PDFs (or TIFF folders) and a metadata CSV,
the pipeline can create the corpus structure and convert sources automatically
via **Step 0**:

```bash
python scripts/run_pipeline.py data/corpus-new \
    --pdf-dir /path/to/scanned-pdfs \
    --csv /path/to/metadata.csv \
    --pipeline-config configs/pipeline_corpus2.yaml \
    --steps 0,1,2,3,4,5,6 --workers 4
```

This will:
1. Create the corpus directory structure (`imgs/`, `charnet/`, `results/`, `dpis/`, `reports/`)
2. Copy the metadata CSV into the corpus folder
3. Convert all PDFs/TIFFs to normalized PNGs at the configured DPI (default 150)
4. Run the remaining pipeline steps

> **Note:** `--pdf-dir` automatically enables Step 0 even if not listed in `--steps`.

### Single document mode

Run steps 1–4 on a single document folder (useful for testing or reprocessing):

```bash
python scripts/run_pipeline.py data/corpus-1/imgs/doc001 --single-doc --steps 1,2,3,4 --workers 4
```

Steps 0, 5, and 6 are automatically skipped in single-doc mode (they require the full corpus).

### Pipeline reports

Every step produces a JSON report in `{corpus}/reports/` with detailed
statistics (page counts, character counts, cluster counts, warnings, etc.).
After all steps complete, a combined `pipeline_report.json` is assembled
automatically.  See [Reports](#reports) for details.

### All CLI options

| Flag | Description |
|------|-------------|
| `--steps` | Comma-separated step numbers to run (default: `1,2,3,4`) |
| `--pipeline-config` | YAML config for detector, segmentation, and conversion settings |
| `--pdf-dir` | Source directory of PDFs/TIFFs for Step 0 |
| `--csv` | Metadata CSV to copy into corpus (used with `--pdf-dir`) |
| `--single-doc` | Process a single document folder instead of full corpus |
| `--workers N` | Parallel workers (default: ncpus-1) |
| `--skip-existing` | Skip documents/pages that already have outputs |
| `--charnet-config` | Override CharNet config file path |
| `--acontrario-config` | Override a contrario YAML config path |
| `--dry-run` | Show what would be run without executing |

---


## Pipeline Overview

The full pipeline has **7 steps** (0–6).  Step 0 is optional — it converts
raw PDFs/TIFFs to PNGs.  Steps 1–6 operate on the extracted images.

### Expected corpus structure

```
data/corpus-1/
  ordered-table-corpus1.csv   # metadata CSV (required)
  imgs/                        # page images (PNG)
    document_001/
      page_0.png
      page_1.png
      ...
    document_002/
      ...
  charnet/                     # OCR + extraction outputs (created by steps 1–4)
  results/                     # a contrario outputs (step 6)
  dpis/                        # precomputed DPI CSVs (optional)
  reports/                     # per-step JSON reports (auto-created)
  distances_roman.npz          # step 5 output
  distances_italic.npz         # step 5 output
```

If you start from PDFs with `--pdf-dir`, the pipeline creates this structure
automatically (see [Quick Start](#quick-start)).

### Corpus Metadata CSV

**Required:** Each corpus folder must contain a metadata CSV file with `corpus` in its filename (e.g. `corpus_known.csv`, `ordered-table-corpus1.csv`).

**Required columns:**
- `Index`: Numbered index for each document (integer, unique per document)
- `Printer`: Printer name for each document (if known; otherwise leave blank or use `unknown`)
- `FileName`: Name of each document (should match the folder or file names exactly)

This file is used for printer attribution and document mapping. If missing, the pipeline will run but printer information will be marked as unknown and some features may be unavailable.

---

### Step 0 — Source Conversion (optional)

Convert scanned PDFs and/or TIFF folders to normalized PNG page images.
This step is only needed when starting from raw scans rather than
pre-extracted images.  It is triggered automatically when `--pdf-dir` is
passed to `run_pipeline.py`, or can be run standalone:

```bash
python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs \
    --dpi-csv-dir data/corpus-1/dpis --workers 8 --skip-existing
```

The original DPI is determined from precomputed CSVs (preferred) or embedded
PDF/TIFF metadata, then images are rescaled to the target DPI (default 150).
Multi-layer PDFs are handled automatically — masks are filtered out and the
largest image per page is selected.

**Output:** `data/corpus-1/imgs/{document}/page_{n}.png`

---

### Step 1 — OCR Detection + Recognition

Detect characters and words using **CharNet** (default) or **DocTR**.

#### CharNet (default)

Make sure the pretrained weights are present in
`src/charnet_src/weights/icdar2015_hourglass88.pth` (see
`src/charnet_src/README.md` for download instructions and a fallback mirror).
A CUDA-capable GPU is recommended; CPU fallback is supported but slow.

```bash
python scripts/run_charnet.py configs/icdar2015_hourglass88.yaml \
    data/corpus-1/imgs  data/corpus-1/charnet \
    --workers 4
```

#### DocTR (alternative)

DocTR uses a detection backbone (e.g. `fast_base`, `db_resnet50`) plus a
custom CRNN recognition head with CTC-based character boundary decoding.
It produces CharNet-compatible JSON so all downstream steps work identically.

```bash
# Standalone
python scripts/run_doctr.py data/corpus-2/imgs data/corpus-2/charnet

# Via pipeline (recommended)
python scripts/run_pipeline.py data/corpus-2 \
    --pipeline-config configs/pipeline_corpus2.yaml --steps 1
```

The pipeline config selects the detector: set `detector: doctr` and
configure `doctr.det_arch` (see [Pipeline Config](#pipeline-config-charnet-vs-doctr)).

**Output:** `data/corpus-1/charnet/{document}/{page}.json` — word/character bounding boxes with recognition scores.

<details>
<summary>Show example output</summary>

![Step 1 — CharNet detections](docs/images/step1_charnet.png)

</details>

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

<details>
<summary>Show example output</summary>

![Step 2 — Extracted characters](docs/images/step2_beforeafter.png)

</details>

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

<details>
<summary>Show example output</summary>

![Step 3 — Italic detection](docs/images/step3_italic.png)

</details>

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

<details>
<summary>Show example output</summary>

![Step 4 — Cluster means](docs/images/step4_clusters.png)

</details>

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

<details>
<summary>Show example output</summary>

![Step 5 — Distance matrix](docs/images/step5_distances.png)

</details>

---

### Step 6 — A Contrario Analysis

NFA-based detection: identify pairs of documents whose typographic similarity is
statistically significant. Produces heatmaps and UMAP similarity graphs.

```bash
python scripts/run_acontrario.py data/corpus-1 \
    --config configs/acontrario_corpus1.yaml
```

**Configuration:** Per-corpus YAML files in `configs/` control analysis parameters,
printer colours, marker shapes, and figure settings.  The `ordering` field
(average/roman/italic/index) selects how books are arranged; `index` simply uses
document indices without reordering by typographic weights.

**Output:** `results/` directory in the corpus root containing:
- `acontrario_results.npz` — n̂₁ matrices, weights, book metadata
- `matrix_round.{png,svg}`, `matrix_cursive.{png,svg}` — heatmaps
- `graph_round.{png,svg}`, `graph_cursive.{png,svg}` — UMAP graphs

<details>
<summary>Show example output</summary>

![Step 6 — A contrario](docs/images/step6_acontrario.png)

</details>

---

## Reports

Every pipeline step writes a JSON report to `{corpus}/reports/`.  When
running via `run_pipeline.py`, all individual reports are merged into a
combined `pipeline_report.json` at the end.

| Report file | Step | Key statistics |
|-------------|------|----------------|
| `convert_sources_report.json` | 0 | Sources processed, pages extracted, DPI, scale factors, warnings on multi-layer PDFs |
| `ocr_detection_report.json` | 1 | Words/characters per page, mean confidence, processing time |
| `character_extraction_report.json` | 2 | Characters extracted per page, segmentation method used |
| `italic_detection_report.json` | 3 | Roman/italic counts, italic ratio, Otsu threshold, stroke quantiles |
| `clustering_report.json` | 4 | Cluster counts (initial/final/roman/italic), mean confidence, top letters |
| `typographic_distances_report.json` | 5 | Documents compared, letters used, mean/median distance per style |
| `acontrario_report.json` | 6 | Significant pairs, n̂₁ statistics, ordering method, output files |
| `pipeline_report.json` | all | Steps run, detector, segmentation, workers, total time, all step reports |

Reports are plain JSON — useful for debugging, logging, and feeding into
downstream dashboards or analysis notebooks.

---

## Notebooks

Interactive notebooks for visualization and debugging:

| Notebook | Description |
|----------|-------------|
| `visualize_char_extraction.ipynb` | Inspect extracted characters, bounding boxes, and orientation data |
| `visualize_italic.ipynb` | Visualize italic/round classification with stroke histograms |
| `visualize_clustering.ipynb` | Browse cluster means, character assignments, and compare documents |
| `visualize_detections.ipynb` | Browse CharNet OCR detections overlaid on page images |
| `acontrario.ipynb` | Interactive a contrario analysis and graph exploration |
| `debug_char_segment.ipynb` | Debug character segmentation algorithm |
| `debug_typographic_distances.ipynb` | Debug distance computation and per-letter analysis |

---

## Utilities

### Catalogue Search

Find PDFs matching catalogue signatures and optionally copy them:

```bash
python scripts/search_pdfs.py catalogue.csv data/pdfs/ --out data/selected/ --skip-existing
```

### DPI Estimation

Compute original DPI and physical page dimensions for PDF or TIFF collections:

```bash
# Single CSV report
python scripts/compute_dpi.py data/corpus-1/pdfs -o dpi_report.csv

# Per-source CSV files (for use with convert_sources.py --dpi-csv-dir)
python scripts/compute_dpi.py data/corpus-1/pdfs -o data/corpus-1/dpis --per-file --skip-existing
```

### Source to PNG Conversion

Convert scanned PDFs and/or TIFF folders to normalized PNG page images at a
target DPI.  The input directory may contain a mix of `.pdf` files and
subfolders of `.tif`/`.tiff` pages — both are auto-detected:

```bash
# Basic conversion (PDFs + TIFF folders)
python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs

# With precomputed DPI values and parallel processing
python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs \
    --dpi-csv-dir data/corpus-1/dpis --workers 8 --skip-existing

# Custom target DPI
python scripts/convert_sources.py data/corpus-1/pdfs data/corpus-1/imgs --target-dpi 300
```

### Validate Pipeline Outputs

```bash
python scripts/validate_outputs.py data/corpus-1/charnet/doc001
python scripts/validate_outputs.py data/corpus-1/charnet --all
```

### Regenerate README Images

```bash
python scripts/generate_readme_images.py
```

---

## Repository Structure

```
scripts/                # Pipeline entry points
  ├── run_pipeline.py           # Master pipeline (steps 0–6)
  ├── convert_sources.py        # Step 0: PDF/TIFF → PNG conversion
  ├── run_charnet.py            # Step 1: CharNet OCR
  ├── run_doctr.py              # Step 1: DocTR OCR (alternative)
  ├── character_extraction.py   # Step 2: MCP Character Extraction
  ├── italic_detection.py       # Step 3: Structure Tensor Italic detection
  ├── clustering.py             # Step 4: Unsupervised Clustering
  ├── typographic_distances.py  # Step 5: Typographic Distance Calculation
  ├── run_acontrario.py         # Step 6: A contrario Analysis
  ├── compute_dpi.py            # Utility: DPI estimation
  ├── search_pdfs.py            # Utility: catalogue search
  └── validate_outputs.py       # Utility: output validation
configs/                # YAML configuration files
  ├── icdar2015_hourglass88.yaml  # CharNet model config
  ├── pipeline_corpus2.yaml       # Pipeline config (DocTR + logit_init)
  ├── typographic_distances.yaml  # Distance computation params
  ├── acontrario_corpus1.yaml     # A contrario (corpus 1)
  └── acontrario_corpus2.yaml     # A contrario (corpus 2)
src/
  ├── acontrario/       # A contrario algorithms + visualization
  ├── charnet_src/      # CharNet OCR module (third-party)
  ├── doctr_src/        # DocTR recognition + CTC decoding
  ├── image_processing/ # Orientation, segmentation, clustering, PDF utilities
  ├── shared/           # Shared tools (reporting)
  └── typ_distances/    # Distance utilities
notebooks/              # Interactive visualization notebooks
docs/images/            # README figures (auto-generated)
data/                   # Input/output data (not tracked)
```

---

## Third-party Code

### CharNet (modified)

`src/charnet_src/` contains a modified copy of
[CharNet](https://github.com/MalongTech/research-charnet) by Malong
Technologies Co., Ltd., used here for Step 1 (character and word detection).

> Linjie Xing, Zhi Tian, Weilin Huang, Matthew R. Scott.
> *Convolutional Character Networks.* ICCV 2019.

The original code is licensed under
[CC-BY-NC-4.0](src/charnet_src/LICENSE) — **non-commercial use only**.
The pretrained weights downloaded via `download_weights.sh` are also subject
to that licence.

Changes made relative to the original are documented in
[src/charnet_src/NOTICE](src/charnet_src/NOTICE) and annotated in each
modified source file. A reference copy of the unmodified upstream source
is kept in `research-charnet-master/` for diffing purposes.

---

## Citation

```bibtex
@inproceedings{sueltas2026,
  title={Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography},
  author={ ...},
  booktitle={ICDAR},
  year={2026}
}
```
