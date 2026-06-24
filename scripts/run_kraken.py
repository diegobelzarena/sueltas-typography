#!/usr/bin/env python
"""Run DocTR detection + recognition on a hierarchy of document images.

Produces **CharNet-compatible JSON** per page so that the downstream
``character_extraction.py`` script can consume it identically.

Each subdirectory of the input folder is treated as a document; all PNG
files within are processed.  The output folder mirrors the structure.

Example:

    python scripts/run_doctr.py data/corpus-2/imgs data/corpus-2/doctr

Architecture
------------
1. **GPU forward** — DocTR ``db_resnet50`` detection + custom CRNN
   recognition with logit access.
2. **CTC decode** — per-word logits are decoded to find character
   boundaries along the horizontal axis.
3. **JSON emit** — each page is written as a flat list of word dicts
   identical to the format produced by ``run_charnet.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import torch
import unicodedata
from doctr.io import DocumentFile
from doctr.models import detection_predictor

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# from doctr_src.predictor import CustomOCRPredictor
# from doctr_src.recognition import create_custom_recognition_predictor
# from doctr_src.utils import get_word_logits, rcnn_positions
from kraken.tasks import SegmentationTaskModel, RecognitionTaskModel
from kraken.configs import SegmentationInferenceConfig, RecognitionInferenceConfig
from shared.tools.metadata import load_skip_pages
from shared.tools.report import StepReport


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_segmentation_and_recognizer(device: str = "cuda",
                    rec_model: str = "catmus-print-fondue-large") -> tuple[SegmentationTaskModel, RecognitionTaskModel]:
    """Build the Kraken segmentation and recognizer with custom recognition."""
    # Load the default segmentation model
    seg_model = SegmentationTaskModel.load_model()
    # Load the custom recognition model
    rec_model = RecognitionTaskModel.load_model(f'./data/models/{rec_model}.mlmodel')

    if device == "cuda":
        seg_model.to(device)
        rec_model.to(device)
    return seg_model, rec_model


def _get_vocab(model: RecognitionTaskModel) -> list[str]:
    """Return the character vocabulary string used by the recognition model."""
    return ['blank'] + list(model.net.codec.c2l.keys())


# ---------------------------------------------------------------------------
# Transform Kraken Lines & Character cuts to Word-level JSON
# ---------------------------------------------------------------------------

def transform_kraken_output(predictions: list[str], char_cuts: np.ndarray, 
                            line_boundary: np.ndarray, confidences: list[float]) -> list[dict]:
    """Transform Kraken character cuts and line information into word-level JSON.

    Parameters
    ----------
    predictions : list of str
        List of predicted characters.
    char_cuts : (num_chars, 4, 2) float array
        Character cuts in page coordinates.
    line_boundary : (num_points, 2) float array
        Boundary coordinates for the line.
    confidences : list of float
        List of confidence scores for each predicted character.

    Returns
    -------
    list of dicts, each with ``tblr`` and ``labels`` keys.
    """
    blank_idx = 0

    # find the pixels that are at top and bottom of the polygon 
    labels = []
    cuts = []
    confs = []

    (l,t),(r,b) = np.min(line_boundary, axis=0), np.max(line_boundary, axis=0)

    fallback_bottom = np.array([[r-1, t+1], 
                                [r-1, b-1]])

    num_preds = len(predictions)
    for i, char in enumerate(predictions):
        is_combining = unicodedata.combining(char) != 0
        
        # Guard against a combining character being the very first character
        if is_combining and labels:
            # Merge with the previous base character
            labels[-1] += char
            confs[-1] = (confs[-1] + confidences[i]) / 2  # Average confidence
            
            # Get bottom points: next char's top points, or the global fallback
            next_points = char_cuts[i+1][:2] if i < num_preds - 1 else fallback_bottom
            cuts[-1][2:] = np.array(next_points)
            
        else:
            # Start a new character
            labels.append(char)
            confs.append(confidences[i])
            
            top_points = np.array(char_cuts[i][:2])
            next_points = char_cuts[i+1][:2] if i < num_preds - 1 else fallback_bottom
            
            # Stack top and bottom points to form a 4-point cut
            cuts.append(np.vstack((top_points, next_points)))

    # Convert list of (4, 2) arrays into a single (N, 4, 2) NumPy array
    cuts_array = np.array(cuts)

    # Vectorized calculation:
    # cuts_array[:, :, 1] gets all Y coordinates. cuts_array[:, :, 0] gets all X coordinates.
    tblrs = np.column_stack((
        cuts_array[:, :, 1].min(axis=1),  # Top    (min Y)
        cuts_array[:, :, 1].max(axis=1),  # Bottom (max Y)
        cuts_array[:, :, 0].min(axis=1),  # Left   (min X)
        cuts_array[:, :, 0].max(axis=1)   # Right  (max X)
    ))

    chars_dict = []
    for tblr, label, cut, confidence in zip(tblrs, labels, cuts, confidences):
        chars_dict.append({
            "tblr": tblr.tolist(),
            "labels": {label: float(confidence)},
            "cut": cut.tolist(),
        })

    return chars_dict


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(seg_model: SegmentationTaskModel, rec_model: RecognitionTaskModel,
                 image_path: str, output_path: str, workers: int = 2, 
                 skip_existing: bool = False) -> tuple[str, dict]:
    """Run detection + recognition on one page, write CharNet-format JSON.

    Returns (status_string, page_info_dict).
    """
    page_info: dict = {"page": os.path.basename(output_path)}
    if skip_existing and os.path.exists(output_path):
        page_info["status"] = "skip"
        return f"SKIP (exists): {output_path}", page_info

    t0 = time.time()
    img = Image.open(image_path)
    img = img.convert('RGB')

    seg_config = SegmentationInferenceConfig()

    segmentation = seg_model.predict(img, seg_config)

    rec_config = RecognitionInferenceConfig(return_logits = False, num_line_workers = workers)

    predictions = rec_model.predict(img, segmentation, rec_config)

    lines_json: list[dict] = []
    line_confs = []
    for idx, line in enumerate(predictions):
        # Absolute pixel bounding box
        baseline = line.baseline
        boundary = line.boundary
        (l,t),(r,b) = np.min(boundary, axis=0), np.max(boundary, axis=0)
        line_tblr = [int(t), int(b), int(l), int(r)]

        text = line.prediction
        if len(text) == 0:
            continue  # skip empty lines
        confidences = line.confidences
        line_confs.append(float(np.mean(confidences)) if confidences else None)

        # Per-character decoding from CRNN logits
        chars: list[dict] = []
        try:
            chars = transform_kraken_output(text, line.cuts, boundary, confidences)
        except Exception:
            pass  # fall back to empty chars

        
        lines_json.append({
            "tblr": line_tblr,
            "baseline": baseline,
            "boundary": boundary,
            "text": text,
            "confidence": line_confs[-1],
            "chars": chars,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lines_json, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    n_chars = sum(len(l.get("chars", [])) for l in lines_json)
    page_info.update(
        status="ok",
        n_words=len(lines_json),
        n_characters=n_chars,
        mean_word_confidence=round(float(np.mean(line_confs)), 4) if line_confs else None,
        min_word_confidence=round(float(np.min(line_confs)), 4) if line_confs else None,
        processing_time=round(elapsed, 3),
    )
    return f"OK: {output_path}  ({len(lines_json)} lines, {elapsed:.2f}s)", page_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Kraken detection + recognition and emit "
                    "CharNet-compatible JSON per page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
        Examples:
        python scripts/run_kraken.py data/corpus-2/imgs data/corpus-2/kraken
        python scripts/run_kraken.py data/corpus-2/imgs data/corpus-2/kraken --skip-existing
        python scripts/run_kraken.py data/corpus-2/imgs/doc001 data/corpus-2/kraken/doc001 --single-doc
                """,
    )
    parser.add_argument("input_root",
                        help="Root folder containing document subfolders "
                             "with PNGs, or a single document folder")
    parser.add_argument("output_root",
                        help="Where to write JSON results (same structure)")
    parser.add_argument("--rec-model", type=str, default="catmus-print-fondue-large",
                        help="Kraken recognition architecture "
                             "(default: catmus-print-fondue-large)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for inference (default: cuda)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pages that already have JSON output")
    parser.add_argument("--single-doc", action="store_true",
                        help="Treat input_root as a single document folder")
    parser.add_argument("--metadata-csv",
                        help="Corpus CSV with a SkipPages column to exclude "
                             "specific pages from processing")
    parser.add_argument("--report-dir",
                        help="Directory for the step report JSON "
                             "(default: {output_root}/../reports)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of worker threads for recognition (default: 2)")  
    args = parser.parse_args(argv)

    # ---- Build model -------------------------------------------------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU.")
        device = "cpu"

    seg_model, rec_model = build_segmentation_and_recognizer(device=device, rec_model=args.rec_model)
    vocab = _get_vocab(rec_model)
    print(f"Vocab length: {len(vocab)}  (CTC blank at index {len(vocab)})")

    # ---- Collect page tasks ------------------------------------------------
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    tasks: list[tuple[str, str]] = []   # (img_path, json_output_path)
    skipped = 0
    skipped_csv = 0

    skip_pages = load_skip_pages(args.metadata_csv) if args.metadata_csv else {}

    if args.single_doc:
        doc_name = input_root.name
        doc_skip = skip_pages.get(doc_name, set())
        for img_file in sorted(input_root.glob("*.png")):
            if img_file.stem in doc_skip:
                skipped_csv += 1
                continue
            out_path = str(output_root / f"{img_file.stem}_lines.json")
            if args.skip_existing and os.path.exists(out_path):
                skipped += 1
                continue
            tasks.append((str(img_file), out_path))
    else:
        for doc_dir in sorted(input_root.iterdir()):
            if not doc_dir.is_dir():
                continue
            doc_skip = skip_pages.get(doc_dir.name, set())
            out_dir = output_root / doc_dir.name
            for img_file in sorted(doc_dir.glob("*.png")):
                if img_file.stem in doc_skip:
                    skipped_csv += 1
                    continue
                out_path = str(out_dir / f"{img_file.stem}_lines.json")
                if args.skip_existing and os.path.exists(out_path):
                    skipped += 1
                    continue
                tasks.append((str(img_file), out_path))

    if skipped_csv:
        print(f"Skipped {skipped_csv} pages via metadata CSV.")
    if skipped:
        print(f"Skipped {skipped} pages with existing outputs.")

    if not tasks:
        print("No pages to process.")
        return 0

    print(f"Processing {len(tasks)} pages on {device}.")

    # ---- Process -----------------------------------------------------------
    report = StepReport("ocr_detection", detector="kraken", rec_model=args.rec_model)
    t_start = time.time()
    done = 0
    for img_path, out_path in tasks:
        msg, page_info = process_page(seg_model, rec_model, img_path, out_path,
                           workers=args.workers, skip_existing=args.skip_existing)
        # Derive doc name from path structure
        doc_name = Path(img_path).parent.name
        page_name = Path(img_path).stem
        report.add_page(doc_name, page_name, page_info)
        done += 1
        if done % 20 == 0 or done == len(tasks):
            print(f"  [{done}/{len(tasks)}] {msg}")

    elapsed = time.time() - t_start
    avg = elapsed / len(tasks) if tasks else 0
    print(f"\nDone: {len(tasks)} pages in {elapsed:.1f}s "
          f"({avg:.2f}s/page avg)")

    # ---- Build report summary & per-doc aggregates -------------------------
    total_words = 0
    total_chars = 0
    for doc_name, doc_data in report.documents.items():
        pages = doc_data.get("pages", {})
        dw = sum(p.get("n_words", 0) for p in pages.values())
        dc = sum(p.get("n_characters", 0) for p in pages.values())
        doc_data["total_pages"] = len(pages)
        doc_data["total_words"] = dw
        doc_data["total_characters"] = dc
        total_words += dw
        total_chars += dc

    report.set_summary({
        "total_documents": len(report.documents),
        "total_pages": len(tasks) + skipped,
        "pages_processed": len(tasks),
        "pages_skipped": skipped,
        "total_words": total_words,
        "total_characters": total_chars,
    })

    # Save report next to the output JSONs
    report_dir = Path(args.report_dir) if args.report_dir else output_root.parent / "reports"
    rpath = report.save(report_dir)
    print(f"Report saved: {rpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
