# Example Data

This folder contains (or should contain) sample documents to verify the pipeline setup.

## Quick Start

1. Download the example document:
   ```bash
   python scripts/download_example.py
   ```

2. Run the pipeline:
   ```bash
   python scripts/run_pipeline.py examples/sample_doc --single-doc --steps 1,2,3,4
   ```

3. Inspect results:
   - Open `notebooks/visualize_char_extraction.ipynb`
   - Set `DOCUMENT = "sample_doc"` and `CHARNET_ROOT = "../examples/charnet"`

## Expected Structure

After running the pipeline, you should have:

```
examples/
├── README.md           (this file)
├── imgs/
│   └── sample_doc/
│       ├── page_001.png
│       └── page_002.png
├── charnet/
│   └── sample_doc/
│       ├── page_001.json         # CharNet detections
│       ├── page_001_data.npz     # Character extraction
│       ├── page_002.json
│       ├── page_002_data.npz
│       ├── italic_labels.npz     # Italic detection
│       └── clusters_all.npz      # Clustering
```

## Manual Setup

If the download script doesn't work, you can manually add any document:

1. Create `examples/imgs/your_doc/` with PNG page images
2. Run: `python scripts/run_pipeline.py examples --steps 1,2,3,4`

## Expected Outputs

For each pipeline step, you should see:

| Step | Output | Expected |
|------|--------|----------|
| 1 | `*.json` | One per page, contains `words[].tblr`, `words[].chars[].tblr` |
| 2 | `*_data.npz` | `char_imgs` (N,40,32), `char_labels` (N,), `word_orientations` |
| 3 | `italic_labels.npz` | `char_italic` (N,), `threshold` |
| 4 | `clusters_all.npz` | `cluster_labels` (N,), `cluster_means` (K,40,32), `cluster_counts` (K,) |

## Validation

Run the validation script to check all outputs exist:
```bash
python scripts/validate_outputs.py examples/charnet/sample_doc
```
