#!/usr/bin/env python
"""Post-process CharNet OCR results: compute orientations, stroke angles,
and character segmentation masks.

For each page that has a CharNet JSON file, this script:

1. Loads the grayscale image and the word/char bounding boxes from JSON.
2. Masks out non-word regions (``filter_image_by_words``, padding=10 px).
3. Runs a sliding-window FFT to estimate local page orientation.
4. Computes the structure tensor on the filtered image.
5. For each word:
   a. Assigns a page orientation (inverse-distance weighted from FFT windows).
   b. Computes a stroke orientation from the structure tensor crop.
   c. Crops the word (with padding = height/5), converts CharNet char tblrs
      to local coordinates, and runs ``char_segment`` to obtain character masks.
6. Saves per-word orientations and refined char tblrs to a ``.npz`` file,
   and character masks to a ``.joblib`` file.

Usage
-----
    python scripts/character_extraction.py \\
        data/corpus-1/imgs  data/corpus-1/charnet  [--workers 4]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the src/ packages are importable even without pip install -e .
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from image_processing.orientation import (
    filter_image_by_words,
    local_fft_sliding_window,
    structure_tensor,
)
from image_processing.char_segment import char_segment


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


def _crop_word(img, t, b, l, r, char_tblrs):
    """Crop the word region with padding = height/5, expanding to contain
    all char bboxes.  Returns (crop, pad_t, pad_l) so that original page
    coordinate ``(py, px)`` maps to crop coordinate
    ``(py - (t - pad_t), px - (l - pad_l))``.
    """
    img_h, img_w = img.shape[:2]
    word_h = b - t
    pad = max(1, word_h // 5)

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

    return img[crop_t:crop_b, crop_l:crop_r], crop_t, crop_l


def _prepare_word_image(word_crop):
    """Contrast-normalise the word crop for char_segment (values in [0,1])."""
    crop_f = word_crop.astype(np.float64)
    kh = max(1, (crop_f.shape[0] // 2) * 2 + 1)
    kw = max(1, (crop_f.shape[1] // 2) * 2 + 1)
    blur = cv2.GaussianBlur(crop_f, (kw, kh),
                            sigmaX=crop_f.shape[1] // 2,
                            sigmaY=crop_f.shape[0] // 2)
    cont = crop_f / (blur + 1e-6)
    cont -= cont.min()
    mx = cont.max()
    if mx > 0:
        cont /= mx
    return cont


# ---------------------------------------------------------------------------
# Core per-page processing function (runs in a worker process).
# ---------------------------------------------------------------------------

def process_page(img_path: str, json_path: str, out_stem: str,
                 window_height: int = 512, window_width: int = 512,
                 step_size: tuple = (256, 256), padding: int = 10) -> str:
    """Process a single page: orientations + char segmentation.

    Writes ``{out_stem}.npz`` and ``{out_stem}.joblib``.
    Returns a status string.
    """
    try:
        # -- Load image (grayscale) ------------------------------------------
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"SKIP (unreadable): {img_path}"
        img_h, img_w = img.shape

        # -- Load CharNet JSON -----------------------------------------------
        with open(json_path) as f:
            words = json.load(f)
        if not words:
            return f"SKIP (no words): {img_path}"

        word_tblrs = [w["tblr"] for w in words if "tblr" in w]

        # -- Step 1: mask non-word regions -----------------------------------
        filtered_img = filter_image_by_words(img, word_tblrs, padding=padding)
        if filtered_img is None:
            return f"SKIP (filter failed): {img_path}"

        # -- Step 2: sliding-window FFT orientation --------------------------
        orientations, positions = local_fft_sliding_window(
            filtered_img,
            window_height=window_height,
            window_width=window_width,
            step_size=step_size,
        )
        if len(orientations) == 0:
            return f"SKIP (no FFT windows): {img_path}"

        # -- Step 3: structure tensor ----------------------------------------
        img_norm = filtered_img.astype(np.float64) / 255.0
        tensor = structure_tensor(img_norm, sigma=0.0, rho=0.3)
        Jxx, Jxy, Jyy = tensor["Jxx"], tensor["Jxy"], tensor["Jyy"]

        # -- Step 4: per-word processing -------------------------------------
        word_orientations = np.full(len(words), np.nan, dtype=np.float32)
        word_stroke_orientations = np.full(len(words), np.nan, dtype=np.float32)

        all_masks = []          # flat list of masks across all words
        word_char_tblrs = []    # one array per word (refined tblrs in page coords)
        word_mask_indices = []  # indices into all_masks for each word

        for wi, word in enumerate(words):
            wt, wb, wl, wr = word["tblr"]
            chars = word.get("chars", [])
            n_chars = len(chars)

            # Centre of the word in page coordinates
            cx = (wl + wr) / 2.0
            cy = (wt + wb) / 2.0

            # 4a. Page orientation for this word
            page_ori = _assign_page_orientation(
                cx, cy, orientations, positions, window_height, window_width)
            word_orientations[wi] = page_ori

            # 4b. Stroke orientation from structure tensor crop
            Jxx_crop = Jxx[wt:wb, wl:wr]
            Jxy_crop = Jxy[wt:wb, wl:wr]
            Jyy_crop = Jyy[wt:wb, wl:wr]
            word_stroke_orientations[wi] = _compute_stroke_orientation(
                Jxx_crop, Jxy_crop, Jyy_crop, page_ori, n_chars)

            # 4c. Character segmentation
            if n_chars < 2:
                word_char_tblrs.append(np.array([]))
                word_mask_indices.append(np.array([], dtype=np.int64))
                continue

            # Gather char tblrs in page coordinates
            char_tblrs_page = np.array([c["tblr"] for c in chars])  # (n, 4)

            # Crop the word region (padding = height/5, expanded for chars)
            crop, crop_t, crop_l = _crop_word(
                img, wt, wb, wl, wr, char_tblrs_page)
            if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
                word_char_tblrs.append(np.array([]))
                word_mask_indices.append(np.array([], dtype=np.int64))
                continue

            # Convert char tblrs to crop-local coordinates
            local_tblrs = char_tblrs_page.copy()
            local_tblrs[:, 0] -= crop_t  # t
            local_tblrs[:, 1] -= crop_t  # b
            local_tblrs[:, 2] -= crop_l  # l
            local_tblrs[:, 3] -= crop_l  # r

            # Clamp to crop bounds
            ch, cw = crop.shape
            local_tblrs[:, 0] = np.clip(local_tblrs[:, 0], 0, ch - 1)
            local_tblrs[:, 1] = np.clip(local_tblrs[:, 1], 1, ch)
            local_tblrs[:, 2] = np.clip(local_tblrs[:, 2], 0, cw - 1)
            local_tblrs[:, 3] = np.clip(local_tblrs[:, 3], 1, cw)

            # Filter out degenerate boxes
            valid = (local_tblrs[:, 1] > local_tblrs[:, 0]) & \
                    (local_tblrs[:, 3] > local_tblrs[:, 2])
            if not valid.any():
                word_char_tblrs.append(np.array([]))
                word_mask_indices.append(np.array([], dtype=np.int64))
                continue
            local_tblrs = local_tblrs[valid]

            # Prepare contrast-normalised image for char_segment
            pre_img = _prepare_word_image(crop)

            widths = local_tblrs[:, 3] - local_tblrs[:, 2]
            refwidth = float(np.mean(widths)) if len(widths) > 0 else 1.0
            box_clu = np.ones(len(local_tblrs), dtype=int)

            # Run char_segment
            try:
                char_data, _ = char_segment(
                    pre_img, local_tblrs.copy(), box_clu, refwidth)
            except Exception:
                char_data = []

            if not char_data:
                word_char_tblrs.append(np.array([]))
                word_mask_indices.append(np.array([], dtype=np.int64))
                continue

            # Convert refined tblrs back to page coordinates and collect masks
            refined_tblrs = []
            mask_idxs = []
            for (rt, rb, rl, rr), mask in char_data:
                refined_tblrs.append([
                    rt + crop_t, rb + crop_t,
                    rl + crop_l, rr + crop_l,
                ])
                all_masks.append(mask)
                mask_idxs.append(len(all_masks) - 1)

            word_char_tblrs.append(np.array(refined_tblrs, dtype=np.int32))
            word_mask_indices.append(np.array(mask_idxs, dtype=np.int64))

        # -- Save results ----------------------------------------------------
        out_dir = os.path.dirname(out_stem)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # .npz: orientations + per-word char tblrs and mask index arrays
        np.savez_compressed(
            f"{out_stem}.npz",
            fft_orientations=orientations,
            fft_positions=positions,
            word_orientations=word_orientations,
            word_stroke_orientations=word_stroke_orientations,
            **{f"char_tblrs_{i}": v for i, v in enumerate(word_char_tblrs)},
            **{f"mask_idx_{i}": v for i, v in enumerate(word_mask_indices)},
        )

        # .joblib: list of boolean masks (one per char across all words)
        joblib.dump(all_masks, f"{out_stem}.joblib", compress=0)

        n_masks = len(all_masks)
        return f"OK: {out_stem}  ({len(words)} words, {n_masks} masks)"

    except Exception as exc:
        import traceback
        return f"ERROR ({img_path}): {exc}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute orientations and character segmentation "
                    "for CharNet-processed document pages.")
    parser.add_argument("image_root",
                        help="Root folder with document subfolders of PNGs")
    parser.add_argument("json_root",
                        help="Root folder with CharNet JSON outputs "
                             "(same subfolder structure)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: ncpus-1)")
    parser.add_argument("--window-height", type=int, default=512)
    parser.add_argument("--window-width", type=int, default=512)
    parser.add_argument("--step-y", type=int, default=256)
    parser.add_argument("--step-x", type=int, default=256)
    parser.add_argument("--padding", type=int, default=10,
                        help="Padding around words for image masking")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pages that already have output files")
    args = parser.parse_args(argv)

    image_root = Path(args.image_root)
    json_root = Path(args.json_root)

    num_workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    step_size = (args.step_y, args.step_x)

    # -- Collect page tasks --------------------------------------------------
    tasks = []  # (img_path, json_path, out_stem)
    for doc_dir in sorted(image_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        json_dir = json_root / doc_dir.name
        if not json_dir.is_dir():
            continue
        for img_file in sorted(doc_dir.glob("*.png")):
            json_file = json_dir / f"{img_file.stem}.json"
            if not json_file.exists():
                continue
            out_stem = str(json_dir / f"{img_file.stem}_orientation")
            if args.skip_existing and (
                    os.path.exists(f"{out_stem}.npz") and
                    os.path.exists(f"{out_stem}.joblib")):
                continue
            tasks.append((str(img_file), str(json_file), out_stem))

    if not tasks:
        print("No pages to process.")
        return

    print(f"Found {len(tasks)} pages.  Using {num_workers} workers.")

    # -- Run -----------------------------------------------------------------
    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                process_page, img_p, json_p, out_s,
                args.window_height, args.window_width, step_size, args.padding,
            )
            for img_p, json_p, out_s in tasks
        ]
        for fut in futures:
            msg = fut.result()
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {msg}")

    print("Done.")


if __name__ == "__main__":
    main()