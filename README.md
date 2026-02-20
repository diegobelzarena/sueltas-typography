# Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography


Official implementation of the paper *Theatre Chapbooks At Scale: A Statistical Comparative Analysis of Typography*, as submitted to ICDAR2026.


## Install

Python 3.11 is recommended.

Install dependencies and make the package importable:

```sh
pip install -r requirements.txt    # install any tools
pip install -e .                   # set up the project in editable mode
```

(After this your `python` interpreter will find the `io` package.)


## Data preparation

The repository expects document images organised as described in the
"Dataset input specification" (see paper or docs). A helper script is
provided for converting a directory of scanned PDFs into the folder
structure used by the pipeline:

```sh
python scripts/convert_pdfs.py /path/to/input_pdfs /path/to/output_images
```

The converter takes two positional arguments: the directory containing
PDFs and the destination directory where each PDF will produce its own
subfolder of PNGs.  Before extraction it looks for a matching CSV in
an optional `--dpi-csv-dir`; that CSV should contain DPI estimates for
each page.  A single DPI value is computed as the median of all
reported DPIs and rounded to the nearest multiple of 50.  The output
images are then rescaled so that their metadata declares a uniform
150 dpi.

Use the `--dpi-csv-dir` option to point the converter at the folder
containing those CSV files.

Example:

```sh
python scripts/convert_pdfs.py --metadata --info pdfs/ imgs/
``````

Each PDF will produce a subdirectory containing a `page_<n>.png` file for
each page.

Another utility script helps match catalogue signatures to actual PDF
files and optionally copy them out of a larger folder:

```sh
python scripts/search_pdfs.py \
    data/corpus-1/ordered-table-corpus1.csv \
    /path/to/pdf_folder \
    --out /path/to/output_dir
```

The `--out` option is optional; when provided the script will copy every
PDF it finds into the target directory, which is useful for assembling a
curated subset.

## Repository layout

- `src/` - Python package and modules for data handling and analysis
- `scripts/` - utility entry‑point scripts (e.g. `convert_pdfs.py`)
- `examples/` - small example datasets or notebooks
- `tests/` – unit tests
- `notebooks/` – exploratory Jupyter notebooks
- `requirements.txt` – global Python dependencies

More detailed documentation is available in the `docs/` directory (if
present) or the paper itself.

## Character network module

Additional code from the CharNet repository has been integrated under
`src/charnet`.  Its dependencies are listed in `requirements.txt`, and
it is installed automatically when the project is installed via
`pip install -e .`.

A convenience script `scripts/run_charnet.py` runs the network across a
folder of documents.  Input should follow the standard layout – each
document has its own subdirectory containing PNG pages – and the output
root will mirror that structure:

```sh
python scripts/run_charnet.py config.yaml input_root/ output_root/
```

Results for each image are saved as **JSON files** containing
character bounding boxes and probability scores.

## DPI / size reporting

A helper module and script can scan a directory of PDFs and/or
subdirectories of TIFFs and emit a consolidated CSV describing each
page’s resolution and physical dimensions.  The logic was borrowed from
an existing standalone script and folded into the package as
``src/io/dpi_info.py``

```sh
python scripts/compute_dpi.py /path/to/root_dir -o dpi.csv
```

Optionally write a CSV for each input file or folder rather than a
single consolidated table:

```sh
python scripts/compute_dpi.py /path/to/root_dir -o output_directory --per-file
```

The script strips any original extension from the output names, so a
source called `foo.pdf` will produce `output_directory/foo.csv`.
The output contains columns for batch, suelta, page filename, DPI
(width/height), dimensions in inches and centimetres, and the source of
those physical values (PDF metadata or TIFF DPI).  This is useful for
quality control and for populating the datasets used in the paper.
