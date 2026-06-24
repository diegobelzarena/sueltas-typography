#!/usr/bin/env python
"""Post-process Kraken OCR results: compute orientations, stroke angles,
character segmentation masks, and embedded character images.

For each page that has a Kraken JSON file, this script:

1. Loads the grayscale image and the word/char bounding boxes from JSON.
2. Applies ``bg_flatten`` to the whole page (Poisson-based background removal).
3. Masks out non-word regions (``filter_image_by_words``, padding=10 px).
4. Runs a sliding-window FFT to estimate local page orientation.
5. Computes the structure tensor on the filtered image.
6. For each word:
   a. Assigns a page orientation (inverse-distance weighted from FFT windows).
   b. Computes a stroke orientation from the structure tensor crop.
   c. Crops the word (with padding = height/5), converts Kraken char tblrs
      to local coordinates, and runs ``char_segment`` to obtain character masks.
   d. Extracts masked character images (bg=1) and embeds them to 40×32.
7. Saves everything to a single ``.npz`` file per page.

Usage
-----
    python scripts/character_extraction.py \\
        data/corpus-1/imgs  data/corpus-1/ocr/kraken  [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable even without pip install -e .
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from image_processing.orientation import (
    filter_image_by_words,
    local_fft_sliding_window,
    structure_tensor,
)
from image_processing.char_segment import char_segment, segment_characters
from image_processing.preprocessing import bg_flatten, embed_noresize
from shared.tools.metadata import load_skip_pages
from shared.tools.report import StepReport


# ---------------------------------------------------------------------------
# Per-word processing helpers
# ---------------------------------------------------------------------------

def _assign_page_orientation(center_x, center_y,
                             orientations, positions,
                             window_height, window_width):
    """Assign a page orientation to a word centre via inverse-distance
    weighted average of overlapping FFT windows."""
    half_h = window_height // 2
    half_w = window_width // 2

    contains = (
        (center_x >= positions[:, 0] - half_w) &
        (center_x <  positions[:, 0] + half_w) &
        (center_y >= positions[:, 1] - half_h) &
        (center_y <  positions[:, 1] + half_h)
    )

    if np.any(contains):
        cp = positions[contains]
        co = orientations[contains]
        dists = np.maximum(
            np.sqrt((cp[:, 0] - center_x) ** 2 +
                    (cp[:, 1] - center_y) ** 2),
            1.0,
        )
        weights = 1.0 / dists
        weights /= weights.sum()
        return float(np.sum(co * weights))

    # Fallback: nearest window
    dists = np.sqrt((positions[:, 0] - center_x) ** 2 +
                    (positions[:, 1] - center_y) ** 2)
    return float(orientations[np.argmin(dists)])


def _compute_stroke_orientation(Jxx_crop, Jxy_crop, Jyy_crop,
                                page_orientation, n_chars):
    """Stroke orientation from the structure tensor crop.

    Returns -360 for single-character words (undefined).
    """
    if n_chars <= 1:
        return -360.0
    return float(
        np.degrees(0.5 * np.arctan2(2 * Jxy_crop.mean(),
                                     Jxx_crop.mean() - Jyy_crop.mean()))
        + page_orientation
    )


def _crop_line_and_mask(img, tblr, boundary, baseline, char_tblrs):
    """Crop the line region with padding = height/5, expanding to contain
    all char bboxes.  Returns (crop, pad_t, pad_l) so that original page
    coordinate ``(py, px)`` maps to crop coordinate
    ``(py - (t - pad_t), px - (l - pad_l))``.
    """
    img_h, img_w = img.shape[:2]
    t,b,l,r = tblr
    line_h = b - t
    pad = max(1, line_h // 5)

    # Expand the crop region to include all char bboxes
    crop_t = t - pad
    crop_b = b + pad
    crop_l = l - pad
    crop_r = r + pad
    for ct, cb, cl, cr in char_tblrs:
        crop_t = min(crop_t, ct)
        crop_b = max(crop_b, cb)
        crop_l = min(crop_l, cl)
        crop_r = max(crop_r, cr)

    # Clamp to image bounds
    crop_t = max(0, crop_t)
    crop_b = min(img_h, crop_b)
    crop_l = max(0, crop_l)
    crop_r = min(img_w, crop_r)


    # Find mask of original boundary
    mask = np.zeros((crop_b - crop_t, crop_r - crop_l), dtype=bool)
    boundary_local = boundary.copy()
    boundary_local[:, 0] -= crop_l
    boundary_local[:, 1] -= crop_t
    baseline_local = baseline.copy()
    baseline_local[:, 0] -= crop_l
    baseline_local[:, 1] -= crop_t
    mask = cv2.fillPoly(mask.astype(np.uint8), [boundary_local], 1).astype(bool)

    return img[crop_t:crop_b, crop_l:crop_r], crop_t, crop_b, crop_l, crop_r, boundary_local, mask, baseline_local

def _extract_char_image(img_flat, tblr, mask):
    """Extract a masked character image from the bg-flattened page.

    Returns a float image where masked-out pixels are set to 1.0 (white).
    """
    t, b, l, r = tblr
    crop = img_flat[t:b, l:r].copy()
    # Apply mask: keep text pixels, set background to 1.0
    mh, mw = mask.shape
    ch, cw = crop.shape
    # Handle possible size mismatch from rounding
    h = min(mh, ch)
    w = min(mw, cw)
    out = np.ones_like(crop)
    out[:h, :w][mask[:h, :w]] = crop[:h, :w][mask[:h, :w]]
    return out


# ---------------------------------------------------------------------------
# Core per-page processing function (runs in a worker process).
# ---------------------------------------------------------------------------

def process_page(img_path: str, json_path: str, out_stem: str,
                 window_height: int = 512, window_width: int = 512,
                 step_size: tuple = (256, 256), padding: int = 10,
                 embed_h: int = 40, embed_w: int = 32,
                 segmentation: str = "kraken_init") -> tuple[str, dict]:
    """Process a single page: orientations + char segmentation + char images.

    Writes ``{out_stem}.npz`` containing all per-page results.
    Returns (status_string, page_info_dict).
    """
    page_info: dict = {"page": os.path.basename(out_stem), "segmentation": segmentation}
    try:
        # -- Load image (grayscale) ------------------------------------------
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            page_info["status"] = "skip_unreadable"
            return f"SKIP (unreadable): {img_path}", page_info
        img_h, img_w = img.shape

        # -- Load Kraken JSON -----------------------------------------------
        with open(json_path, encoding="utf-8") as f:
            lines = json.load(f)
        if not lines:
            page_info["status"] = "skip_no_lines"
            return f"SKIP (no lines): {img_path}", page_info

        lines_tblrs = [l["tblr"] for l in lines if "tblr" in l]

        # -- Step 1: bg_flatten the whole page -------------------------------
        img_flat = bg_flatten(img, d=3, equalize=True)  # float

        # -- Step 2: mask non-word regions (on uint8 for FFT) ----------------
        filtered_img = filter_image_by_words(img, lines_tblrs, padding=padding)
        if filtered_img is None:
            page_info["status"] = "skip_filter_failed"
            return f"SKIP (filter failed): {img_path}", page_info

        # -- Step 3: sliding-window FFT orientation --------------------------
        orientations, positions = local_fft_sliding_window(
            filtered_img,
            window_height=window_height,
            window_width=window_width,
            step_size=step_size,
        )
        if len(orientations) == 0:
            page_info["status"] = "skip_no_fft"
            return f"SKIP (no FFT windows): {img_path}", page_info

        # -- Step 4: structure tensor ----------------------------------------
        img_norm = filtered_img.astype(np.float64) / 255.0
        tensor = structure_tensor(img_norm, sigma=0.0, rho=0.3)
        Jxx, Jxy, Jyy = tensor["Jxx"], tensor["Jxy"], tensor["Jyy"]

        # -- Step 5: per-word processing -------------------------------------
        # word_orientations = np.full(len(words), np.nan, dtype=np.float32)
        # word_stroke_orientations = np.full(len(words), np.nan, dtype=np.float32)
        word_orientations = []
        word_stroke_orientations = []

        all_char_imgs = []      # raw masked crops (variable size), for embedding
        all_char_labels = []    # OCR label per char
        all_char_word_idx = []  # which word each char belongs to
        all_masks = []          # boolean masks
        word_char_tblrs = []    # one array per word (refined tblrs in page coords)
        word_mask_indices = []  # indices into all_masks for each word
        words_json = []            # JSON output for each word

        for li, line in enumerate(lines):
            lt, lb, ll, lr = line["tblr"]
            line_boundary = np.array(line["boundary"])
            line_baseline = np.array(line["baseline"])
            chars = line.get("chars", [])
            n_chars = len(chars)

            # 5a. Character segmentation
            if n_chars < 2:
                # word_char_tblrs.append(np.array([]))
                # word_mask_indices.append(np.array([], dtype=np.int64))
                continue

            # Gather char tblrs in page coordinates
            char_tblrs_page = np.array([c["tblr"] for c in chars])  # (n, 4)

            # Crop the line region (padding = height/5, expanded for chars)
            # Grayscale crop for path finding, flattened crop for mask filtering
            crop_flat, crop_t, crop_b, crop_l, crop_r, crop_boundary, crop_mask, crop_baseline = _crop_line_and_mask(
                img_flat, line["tblr"], line_boundary, line_baseline, char_tblrs_page)
            crop_gray = img[crop_t:crop_t + crop_flat.shape[0],
                            crop_l:crop_l + crop_flat.shape[1]].copy()
            
            pad_t = lt - crop_t 
            pad_b = crop_b - lb
            pad_l = ll - crop_l
            pad_r = crop_r - lr

            if crop_flat.size == 0 or crop_flat.shape[0] < 3 or crop_flat.shape[1] < 3:
                # word_char_tblrs.append(np.array([]))
                # word_mask_indices.append(np.array([], dtype=np.int64))
                continue

            # Normalize crops to [0, 1] for char_segment
            # Grayscale crop: used for path finding (full contrast info)
            gray_min = crop_gray.min() if crop_gray.size > 0 else 0
            gray_max = crop_gray.max() if crop_gray.size > 0 else 1
            crop_gray_norm = (crop_gray - gray_min) / (gray_max - gray_min + 1e-8)
            # Flattened crop: used for mask filtering (clean background)
            flat_min = crop_flat.min() if crop_flat.size > 0 else 0
            flat_max = crop_flat.max() if crop_flat.size > 0 else 1
            crop_flat_norm = (crop_flat - flat_min) / (flat_max - flat_min + 1e-8)

            # Convert char tblrs to crop-local coordinates
            local_tblrs = char_tblrs_page.copy()
            local_tblrs[:, 0] -= crop_t  # t
            local_tblrs[:, 1] -= crop_t  # b
            local_tblrs[:, 2] -= crop_l  # l
            local_tblrs[:, 3] -= crop_l  # r

            # Clamp to crop bounds
            ch, cw = crop_gray_norm.shape
            local_tblrs[:, 0] = np.clip(local_tblrs[:, 0], 0, ch - 1)
            local_tblrs[:, 1] = np.clip(local_tblrs[:, 1], 1, ch)
            local_tblrs[:, 2] = np.clip(local_tblrs[:, 2], 0, cw - 1)
            local_tblrs[:, 3] = np.clip(local_tblrs[:, 3], 1, cw)

            # Filter out degenerate boxes
            valid = (local_tblrs[:, 1] > local_tblrs[:, 0]) & \
                    (local_tblrs[:, 3] > local_tblrs[:, 2])
            if not valid.any():
                # word_char_tblrs.append(np.array([]))
                # word_mask_indices.append(np.array([], dtype=np.int64))
                continue
            valid_indices = np.where(valid)[0]
            local_tblrs = local_tblrs[valid]

            chars_n = [c['labels'] for c in line['chars']]
            chars_labels = np.array([list(c.keys())[0] for c in chars_n])[valid]
            chars_confs = np.array([list(c.values())[0] for c in chars_n])[valid]
            len_no_sp = len([c for c in chars_labels if c != ' '])

                         # ... (previous code remains the same) ...
            
            # Run character segmentation
            try:
                char_data, _, _ = segment_characters(
                    crop_gray_norm, local_tblrs.copy(),
                    mode="kraken_init",
                    top_pad=pad_t, bottom_pad=pad_b,
                    img_flat=crop_flat_norm, boundary=crop_boundary,
                    baseline=crop_baseline, chars_labels=chars_labels, len_no_sp=len_no_sp)
            except Exception:
                char_data = None

            if not char_data:
                # word_char_tblrs.append(np.array([]))
                # word_mask_indices.append(np.array([], dtype=np.int64))
                continue
        
            # Convert refined tblrs back to page coordinates and collect masks
            refined_tblrs = []
            mask_idxs = []
            w_confidences = []
            w_text = []
            w_chars = []
            
            # FIX 1: Zip all three arrays to prevent index misalignment
            for c_data, label, conf in zip(char_data, chars_labels, chars_confs):
                if (c_data is None) or (label == ' '):
                    if label == ' ':
                        if len(refined_tblrs) == 0:
                            refined_tblrs = []
                            mask_idxs = []
                            w_chars = []
                            w_confidences = []
                            w_text = []
                            continue
                        else:
                            word_char_tblrs.append(np.array(refined_tblrs, dtype=np.int32))
                            word_mask_indices.append(np.array(mask_idxs, dtype=np.int64))
                            
                            refined_tblrs_np = np.array(refined_tblrs)
                            wl, wt = refined_tblrs_np[:, 2].min(), refined_tblrs_np[:, 0].min()
                            wr, wb = refined_tblrs_np[:, 3].max(), refined_tblrs_np[:, 1].max()
                            
                            cx = (wl + wr) / 2.0
                            cy = (wt + wb) / 2.0

                            page_ori = _assign_page_orientation(
                                cx, cy, orientations, positions, window_height, window_width)
                            word_orientations.append(page_ori)

                            Jxx_crop = Jxx[wt:wb, wl:wr]
                            Jxy_crop = Jxy[wt:wb, wl:wr]
                            Jyy_crop = Jyy[wt:wb, wl:wr]
                            
                            # FIX 2: Pass the actual word length, not the line length
                            word_stroke_orientations.append(_compute_stroke_orientation(
                                Jxx_crop, Jxy_crop, Jyy_crop, page_ori, len(w_chars)))
                            
                            words_json.append({
                                "tblr": [int(wt), int(wb), int(wl), int(wr)],
                                "text": ''.join(w_text),
                                "text_score": float(np.mean(w_confidences)) if w_confidences else 0.0,
                                "chars": w_chars,
                            }) 

                            refined_tblrs = []
                            mask_idxs = []
                            w_chars = []
                            w_confidences = []
                            w_text = []
                    continue
                    
                ((rt, rb, rl, rr), mask) = c_data
                page_tblr = [
                    rt + crop_t, rb + crop_t,
                    rl + crop_l, rr + crop_l,
                ]
                refined_tblrs.append(page_tblr)
                all_masks.append(mask)
                mask_idxs.append(len(all_masks) - 1)
                
                # FIX 1 continued: Use the zipped confidence
                w_confidences.append(conf) 
                w_text.append(label)
                w_chars.append({
                    "tblr": [int(page_tblr[0]), int(page_tblr[1]), int(page_tblr[2]), int(page_tblr[3])],
                    "labels": {label: float(conf)},
                })

                char_img = _extract_char_image(img_flat, page_tblr, mask)
                all_char_imgs.append(char_img)
                all_char_labels.append(label)
                all_char_word_idx.append(len(words_json))
                
            if len(refined_tblrs) > 0:
                word_char_tblrs.append(np.array(refined_tblrs, dtype=np.int32))
                word_mask_indices.append(np.array(mask_idxs, dtype=np.int64))
                
                refined_tblrs_np = np.array(refined_tblrs)
                wl, wt = refined_tblrs_np[:, 2].min(), refined_tblrs_np[:, 0].min()
                wr, wb = refined_tblrs_np[:, 3].max(), refined_tblrs_np[:, 1].max()
                
                cx = (wl + wr) / 2.0
                cy = (wt + wb) / 2.0

                page_ori = _assign_page_orientation(
                    cx, cy, orientations, positions, window_height, window_width)
                word_orientations.append(page_ori)

                Jxx_crop = Jxx[wt:wb, wl:wr]
                Jxy_crop = Jxy[wt:wb, wl:wr]
                Jyy_crop = Jyy[wt:wb, wl:wr]
                
                # FIX 2 continued: Pass the actual word length
                word_stroke_orientations.append(_compute_stroke_orientation(
                    Jxx_crop, Jxy_crop, Jyy_crop, page_ori, len(w_chars)))
                
                words_json.append({
                    "tblr": [int(wt), int(wb), int(wl), int(wr)],
                    "text": ''.join(w_text),
                    "text_score": float(np.mean(w_confidences)) if w_confidences else 0.0,
                    "chars": w_chars,
                })
        
        # -- Embed character images to fixed size ----------------------------
        n_chars_total = len(all_char_imgs)
        if n_chars_total > 0:
            char_imgs_embedded, _ = embed_noresize(
                all_char_imgs, h=embed_h, w=embed_w)
        else:
            char_imgs_embedded = np.zeros((0, embed_h, embed_w),
                                         dtype=np.float64)
        
        # -- Save results (single .npz per page) ----------------------------
        out_dir = os.path.dirname(out_stem)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        save_dict = dict(
            # Page-level
            page_height=np.array([img_h]),
            fft_orientations=orientations,
            fft_positions=positions,
            # Per-word
            word_orientations=word_orientations,
            word_stroke_orientations=word_stroke_orientations,
            # Per-character (flat arrays, aligned by index)
            char_imgs=char_imgs_embedded.astype(np.float32),
            char_labels=np.array(all_char_labels, dtype="U1"),
            char_word_idx=np.array(all_char_word_idx, dtype=np.int32),
        )
        # Per-word variable-length arrays
        for i, v in enumerate(word_char_tblrs):
            save_dict[f"char_tblrs_{i}"] = v
        for i, v in enumerate(word_mask_indices):
            save_dict[f"mask_idx_{i}"] = v

        np.savez_compressed(f"{out_stem}.npz", **save_dict)

        # -- Save Word JSON output (one file per page) --------------------------------
        words_json_path = json_path.replace("_lines.json", ".json")
        with open(words_json_path, "w", encoding="utf-8") as f:
            json.dump(words_json, f, indent=2, ensure_ascii=False)

        n_words_with_chars = sum(
            1 for v in word_char_tblrs if len(v) > 0
        )
        page_info.update(
            status="ok",
            n_words=len(words_json),
            n_words_with_chars=n_words_with_chars,
            n_characters_extracted=n_chars_total,
        )
        return f"OK: {out_stem}  ({len(words_json)} words, {n_chars_total} chars)", page_info

    except Exception as exc:
        import traceback
        page_info["status"] = "error"
        page_info["error"] = str(exc)
        return f"ERROR ({img_path}): {exc}\n{traceback.format_exc()}", page_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute orientations and character segmentation "
                    "for Kraken-processed document pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/character_extraction_kraken.py data/corpus-1/imgs data/corpus-1/ocr/kraken
  python scripts/character_extraction_kraken.py data/corpus-1/imgs data/corpus-1/ocr/kraken --workers 4
  python scripts/character_extraction_kraken.py data/corpus-1/imgs/doc001 data/corpus-1/ocr/kraken/doc001 --single-doc
        """,
    )
    parser.add_argument("image_root",
                        help="Root folder with document subfolders of PNGs or a single document folder")
    parser.add_argument("json_root",
                        help="Root folder with Kraken JSON outputs (same subfolder structure or single folder)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: ncpus-1)")
    parser.add_argument("--window-height", type=int, default=512,
                        help="FFT window height in pixels (default: 512)")
    parser.add_argument("--window-width", type=int, default=512,
                        help="FFT window width in pixels (default: 512)")
    parser.add_argument("--step-y", type=int, default=256,
                        help="Vertical step for sliding FFT window (default: 256)")
    parser.add_argument("--step-x", type=int, default=256,
                        help="Horizontal step for sliding FFT window (default: 256)")
    parser.add_argument("--padding", type=int, default=10,
                        help="Padding around words for image masking")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pages that already have output files")
    parser.add_argument("--embed-h", type=int, default=40,
                        help="Height of embedded char images (default: 40)")
    parser.add_argument("--embed-w", type=int, default=32,
                        help="Width of embedded char images (default: 32)")
    parser.add_argument("--single-doc", action="store_true",
                        help="Process a single folder of PNGs and JSONs as one document")
    parser.add_argument("--segmentation", type=str, default="kraken_init",
                        choices=["kraken_init"],
                        help="Segmentation mode: kraken_init (default)")
    parser.add_argument("--metadata-csv",
                        help="Corpus CSV with a SkipPages column to exclude "
                             "specific pages from processing")
    parser.add_argument("--report-dir",
                        help="Directory for the step report JSON "
                             "(default: {json_root}/../reports)")
    args = parser.parse_args(argv)

    image_root = Path(args.image_root)
    json_root = Path(args.json_root)

    num_workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    step_size = (args.step_y, args.step_x)

    # -- Collect page tasks --------------------------------------------------
    tasks = []  # (img_path, json_path, out_stem)
    skipped_csv = 0

    skip_pages = load_skip_pages(args.metadata_csv) if args.metadata_csv else {}

    if args.single_doc:
        # Treat image_root and json_root as single document folders
        doc_skip = skip_pages.get(image_root.name, set())
        for img_file in sorted(image_root.glob("*.png")):
            if img_file.stem in doc_skip:
                skipped_csv += 1
                continue
            json_file = json_root / f"{img_file.stem}_lines.json"
            if not json_file.exists():
                continue
            out_stem = str(json_root / f"{img_file.stem}_data")
            if args.skip_existing and os.path.exists(f"{out_stem}.npz"):
                continue
            tasks.append((str(img_file), str(json_file), out_stem))
    else:
        for doc_dir in sorted(image_root.iterdir()):
            if not doc_dir.is_dir():
                continue
            json_dir = json_root / doc_dir.name
            if not json_dir.is_dir():
                continue
            doc_skip = skip_pages.get(doc_dir.name, set())
            for img_file in sorted(doc_dir.glob("*.png")):
                if img_file.stem in doc_skip:
                    skipped_csv += 1
                    continue
                json_file = json_dir / f"{img_file.stem}_lines.json"
                if not json_file.exists():
                    continue
                out_stem = str(json_dir / f"{img_file.stem}_data")
                if args.skip_existing and os.path.exists(f"{out_stem}.npz"):
                    continue
                tasks.append((str(img_file), str(json_file), out_stem))

    if skipped_csv:
        print(f"Skipped {skipped_csv} pages via metadata CSV.")

    if not tasks:
        print("No pages to process.")
        return

    print(f"Found {len(tasks)} pages.  Using {num_workers} workers.")

    # -- Run -----------------------------------------------------------------
    report = StepReport("character_extraction", segmentation=args.segmentation)
    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                process_page, img_p, json_p, out_s,
                args.window_height, args.window_width, step_size, args.padding,
                args.embed_h, args.embed_w, args.segmentation,
            )
            for img_p, json_p, out_s in tasks
        ]
        for i, fut in enumerate(futures):
            msg, page_info = fut.result()
            img_p = tasks[i][0]
            doc_name = Path(img_p).parent.name
            page_name = Path(img_p).stem
            report.add_page(doc_name, page_name, page_info)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {msg}")

    # -- Per-doc aggregates & summary ----------------------------------------
    total_words = 0
    total_chars = 0
    for doc_name, doc_data in report.documents.items():
        pages = doc_data.get("pages", {})
        dw = sum(p.get("n_words", 0) for p in pages.values())
        dc = sum(p.get("n_characters_extracted", 0) for p in pages.values())
        doc_data["total_pages"] = len(pages)
        doc_data["total_words"] = dw
        doc_data["total_characters_extracted"] = dc
        total_words += dw
        total_chars += dc

    report.set_summary({
        "total_documents": len(report.documents),
        "total_pages": len(tasks),
        "total_words": total_words,
        "total_characters_extracted": total_chars,
    })

    report_dir = Path(args.report_dir) if args.report_dir else json_root.parent / "reports"
    rpath = report.save(report_dir)
    print(f"Report saved: {rpath}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())