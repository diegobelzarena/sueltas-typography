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

import cv2
import numpy as np
import torch
from doctr.io import DocumentFile
from doctr.models import detection_predictor

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from doctr_src.predictor import CustomOCRPredictor
from doctr_src.recognition import create_custom_recognition_predictor
from doctr_src.utils import get_word_logits, rcnn_positions
from shared.tools.metadata import load_skip_pages
from shared.tools.report import StepReport


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_predictor(device: str = "cuda",
                    det_arch: str = "db_resnet50") -> CustomOCRPredictor:
    """Build the DocTR OCR predictor with custom recognition."""
    det_pred = detection_predictor(
        det_arch, pretrained=True,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
    )
    reco_pred = create_custom_recognition_predictor(
        arch="crnn_vgg16_bn", pretrained=True,
        batch_size=32,
        dynamic_width_batching=True,
        split_wide_crops=False,
    )
    predictor = CustomOCRPredictor(
        det_predictor=det_pred,
        reco_predictor=reco_pred,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
    )
    if device == "cuda":
        predictor.to(device)
    return predictor


def _get_vocab(predictor: CustomOCRPredictor) -> str:
    """Return the character vocabulary string used by the recognition model."""
    return predictor.reco_predictor.model.cfg["vocab"]


# ---------------------------------------------------------------------------
# CTC decode → CharNet-compatible char dicts
# ---------------------------------------------------------------------------

def _ctc_decode_chars(logits_word: np.ndarray, word_tblr: list[int],
                      vocab: str, resize_word: np.ndarray,
                      pad_word: np.ndarray, pad_w: int) -> list[dict]:
    """Decode CRNN logits into per-character tblrs and label dicts.

    Uses :func:`rcnn_positions` to properly account for CRNN preprocessing
    (resize scales and padding) when mapping CTC sequence positions to
    pixel coordinates.

    Parameters
    ----------
    logits_word : (seq_len, num_classes) float array
        Raw logits from the CRNN for this word.
    word_tblr : [top, bottom, left, right]
        Absolute pixel bounding box of the word in page coordinates.
    vocab : str
        Character vocabulary (index i → vocab[i]).  The CTC blank class
        sits at index ``len(vocab)``.
    resize_word : array-like, shape (2,)
        ``(scale_h, scale_w)`` applied during CRNN preprocessing.
    pad_word : array-like, shape (2,)
        ``(pad_left, pad_right)`` applied during CRNN preprocessing.
    pad_w : int
        Padding added around the word crop (``word_height // 5``).

    Returns
    -------
    list of dicts, each with ``tblr`` and ``labels`` keys.
    """
    blank_idx = len(vocab)
    wt, wb, wl, wr = word_tblr
    word_h = wb - wt
    word_w = wr - wl
    max_idx = np.argmax(logits_word, axis=1)

    # Collect (seq_position, class_index) for each decoded character
    char_positions: list[tuple[int, int]] = []
    prev = -1
    for s, cls in enumerate(max_idx):
        if cls != blank_idx and cls != prev:
            char_positions.append((s, int(cls)))
        prev = int(cls)

    if not char_positions:
        return []

    # Use rcnn_positions for proper coordinate mapping
    delta_x1 = -pad_w
    delta_x2 = pad_w
    delta_y1 = -pad_w
    delta_y2 = pad_w
    word_img_shape = (word_h + 2 * pad_w, word_w + 2 * pad_w)

    pos_ctc = rcnn_positions(
        logits_word, word_img_shape, resize_word, pad_word,
        delta_y1, delta_y2, delta_x1, delta_x2,
    )

    # Convert padded-crop x-positions to page coordinates
    crop_origin_x = wl - pad_w
    n_chars = len(pos_ctc) - 1

    chars: list[dict] = []
    for ci in range(n_chars):
        char_l = pos_ctc[ci] + crop_origin_x
        char_r = pos_ctc[ci + 1] - 1 + crop_origin_x

        # Softmax over character classes (exclude CTC blank)
        if ci < len(char_positions):
            seq_pos = char_positions[ci][0]
        else:
            seq_pos = 0
        logit_slice = logits_word[seq_pos, :blank_idx].astype(np.float64)
        logit_slice -= logit_slice.max()
        exp_l = np.exp(logit_slice)
        probs = exp_l / exp_l.sum()

        # Top-10 labels by probability
        top10 = np.argsort(probs)[::-1][:10]
        labels = {}
        for idx in top10:
            if idx < len(vocab):
                labels[vocab[idx]] = float(probs[idx])

        chars.append({
            "tblr": [wt, wb, max(int(char_l), wl), min(int(char_r), wr)],
            "labels": labels,
        })
    return chars


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(predictor: CustomOCRPredictor,
                 image_path: str, output_path: str,
                 vocab: str, skip_existing: bool = False) -> tuple[str, dict]:
    """Run detection + recognition on one page, write CharNet-format JSON.

    Returns (status_string, page_info_dict).
    """
    page_info: dict = {"page": os.path.basename(output_path)}
    if skip_existing and os.path.exists(output_path):
        page_info["status"] = "skip"
        return f"SKIP (exists): {output_path}", page_info

    t0 = time.time()
    doc = DocumentFile.from_images(image_path)
    img_h, img_w = doc[0].shape[:2]

    result_dict, logits_data = predictor(
        doc,
        return_intermediate=True,
        reco_kwargs={"return_model_output": True},
    )

    logits = logits_data.get("reco_logits")
    metadata = logits_data.get("batch_metadata")

    if logits is None or metadata is None:
        # No words detected — write empty JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        page_info.update(status="ok", n_words=0, n_characters=0,
                         processing_time=round(time.time() - t0, 3))
        return f"OK (0 words): {output_path}", page_info

    # Make metadata arrays contiguous for safe indexing
    o_idxs = np.ascontiguousarray(metadata["original_indices"])
    r_scls = np.ascontiguousarray(metadata["resize_scales"])
    pads   = np.ascontiguousarray(metadata["padding"])
    b_idxs = np.ascontiguousarray(metadata["batch_indices"])

    words_json: list[dict] = []
    for idx, word in enumerate(result_dict["words"]):
        # Absolute pixel bounding box
        geo = word["geometry_pixel"]
        xs = [p[0] for p in geo]
        ys = [p[1] for p in geo]
        t, b, l, r = int(min(ys)), int(max(ys)), int(min(xs)), int(max(xs))
        word_tblr = [t, b, l, r]

        text = word["text"]
        confidence = word["confidence"]

        # Per-character decoding from CRNN logits
        chars: list[dict] = []
        try:
            word_height = b - t
            logits_word, resize_word, pad_word, pad_w = get_word_logits(
                word_height, logits, o_idxs, r_scls, pads, b_idxs, idx,
            )
            chars = _ctc_decode_chars(logits_word, word_tblr, vocab,
                                      resize_word, pad_word, pad_w)
        except Exception:
            pass  # fall back to empty chars

        words_json.append({
            "tblr": word_tblr,
            "text": text,
            "text_score": float(confidence),
            "word_bbox_score": float(confidence),
            "chars": chars,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(words_json, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    n_chars = sum(len(w.get("chars", [])) for w in words_json)
    word_confs = [w["text_score"] for w in words_json]
    page_info.update(
        status="ok",
        n_words=len(words_json),
        n_characters=n_chars,
        mean_word_confidence=round(float(np.mean(word_confs)), 4) if word_confs else None,
        min_word_confidence=round(float(np.min(word_confs)), 4) if word_confs else None,
        processing_time=round(elapsed, 3),
    )
    return f"OK: {output_path}  ({len(words_json)} words, {elapsed:.2f}s)", page_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run DocTR detection + recognition and emit "
                    "CharNet-compatible JSON per page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_doctr.py data/corpus-2/imgs data/corpus-2/doctr
  python scripts/run_doctr.py data/corpus-2/imgs data/corpus-2/doctr --skip-existing
  python scripts/run_doctr.py data/corpus-2/imgs/doc001 data/corpus-2/doctr/doc001 --single-doc
        """,
    )
    parser.add_argument("input_root",
                        help="Root folder containing document subfolders "
                             "with PNGs, or a single document folder")
    parser.add_argument("output_root",
                        help="Where to write JSON results (same structure)")
    parser.add_argument("--det-arch", type=str, default="db_resnet50",
                        help="DocTR detection architecture "
                             "(default: db_resnet50)")
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
    args = parser.parse_args(argv)

    # ---- Build model -------------------------------------------------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU.")
        device = "cpu"

    predictor = build_predictor(device=device, det_arch=args.det_arch)
    vocab = _get_vocab(predictor)
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
            out_path = str(output_root / f"{img_file.stem}.json")
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
                out_path = str(out_dir / f"{img_file.stem}.json")
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
    report = StepReport("ocr_detection", detector="doctr", det_arch=args.det_arch)
    t_start = time.time()
    done = 0
    for img_path, out_path in tasks:
        msg, page_info = process_page(predictor, img_path, out_path, vocab,
                           skip_existing=args.skip_existing)
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
