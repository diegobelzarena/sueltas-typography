#!/usr/bin/env python
"""Run CharNet on a hierarchy of document images.

Each subdirectory of the input folder is treated as a document; all PNG
files within are processed.  The output folder mirrors the structure,
placing recognition results alongside the images.

Example:

    python scripts/run_charnet.py config.yaml input_root/ output_root/

This is a thin wrapper around the existing ``charnet`` machinery (see
``src/charnet_src/tools/test_net.py``).

Architecture
------------
The script pipelines three stages so that 1 GPU + N CPUs are kept busy:

1. **Image prefetch** — a background thread decodes PNGs from disk and
   resizes them, so the next image is ready by the time the GPU finishes.
2. **GPU forward** — the main thread runs the backbone + heads on the
   GPU, immediately getting back numpy prediction maps.
3. **CPU postprocessing + save** — a ``ProcessPoolExecutor`` (up to 10
   workers) handles NMS, char-word matching, lexicon filtering, and JSON
   serialisation in parallel for different images, while the GPU is
   already working on the next frame.
"""

import os
import argparse
import json
import torch
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
from charnet.config import cfg
from charnet.modeling.model import CharNet
from charnet.modeling.postprocessing import OrientedTextPostProcessing, load_char_dict


# ---------------------------------------------------------------------------
# Helpers that run in *worker processes* (no access to cfg or the model).
# ---------------------------------------------------------------------------

def _postprocess_and_save(preds, scale_w, scale_h, orig_w, orig_h,
                          image_id, save_root, char_dict_file,
                          postproc_kwargs):
    """CPU-only: run postprocessing + save JSON.  Executed in a worker."""
    postproc = _get_postprocessor(postproc_kwargs, char_dict_file)

    char_bboxes, char_scores, word_instances = postproc(
        preds["pred_word_fg"], preds["pred_word_tblr"],
        preds["pred_word_orient"], preds["pred_char_fg"],
        preds["pred_char_tblr"], preds["pred_char_cls"],
        scale_w, scale_h, orig_w, orig_h,
    )

    _save_json(char_bboxes, char_scores, word_instances, image_id,
               save_root, char_dict_file)


# Per-worker cache so we don't rebuild the postprocessor every call.
_worker_postproc = None
_worker_pp_key = None


def _get_postprocessor(kwargs, char_dict_file):
    global _worker_postproc, _worker_pp_key
    key = id(kwargs)  # same dict object → same config
    if _worker_postproc is None or _worker_pp_key != key:
        _worker_postproc = OrientedTextPostProcessing(**kwargs)
        _worker_pp_key = key
    return _worker_postproc


# Per-worker char_dict cache.
_worker_char_dict = None
_worker_cdf = None


def _get_char_dict(char_dict_file):
    global _worker_char_dict, _worker_cdf
    if _worker_char_dict is None or _worker_cdf != char_dict_file:
        _worker_char_dict = load_char_dict(char_dict_file)
        _worker_cdf = char_dict_file
    return _worker_char_dict


def _save_json(char_bboxes, char_scores, word_instances, image_id,
               save_root, char_dict_file):
    char_dict = _get_char_dict(char_dict_file)
    char_ids = list(char_dict.keys())

    def _make_char_det(char_bbox, char_score):
        l, r = int(min(char_bbox[:8:2])), int(max(char_bbox[:8:2]))
        t, b = int(min(char_bbox[1:8:2])), int(max(char_bbox[1:8:2]))
        sorted_ids = sorted(char_ids, key=lambda i: char_score[i],
                            reverse=True)
        return {
            'tblr': [t, b, l, r],
            'polygon': [int(v) for v in char_bbox[:8]],
            'labels': {char_dict[i].lower(): float(char_score[i])
                       for i in sorted_ids},
        }

    # --- word-level detections (with nested chars) ---
    words = []
    for wi in word_instances:
        wb = wi.word_bbox  # 8-element array: x1,y1,...,x4,y4
        l, r = int(min(wb[0:8:2])), int(max(wb[0:8:2]))
        t, b = int(min(wb[1:8:2])), int(max(wb[1:8:2]))
        chars = [_make_char_det(cb, cs)
                 for cb, cs in zip(wi.char_bboxes, wi.char_scores)]
        words.append({
            'tblr': [t, b, l, r],
            'polygon': [int(v) for v in wb[:8]],
            'text': wi.text,
            'text_score': float(wi.text_score),
            'word_bbox_score': float(wi.word_bbox_score),
            'chars': chars,
        })

    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, f"{image_id}.json"), "w") as f:
        json.dump(words, f, indent=2)


# ---------------------------------------------------------------------------
# Image loading (runs in a background thread).
# ---------------------------------------------------------------------------

def resize(im, input_size, size_divisibility):
    """Resize so that the height equals *input_size* (rounded to size_divisibility)."""
    h, w, _ = im.shape
    scale = h / float(input_size)
    image_resize_height = int(round(h / scale / size_divisibility) * size_divisibility)
    image_resize_width = int(round(w / scale / size_divisibility) * size_divisibility)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height),
                    interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def _load_and_resize(path, input_size, size_divisibility):
    """Read an image from *path*, resize, and return all scale info."""
    im_original = cv2.imread(path)
    im, scale_w, scale_h, orig_w, orig_h = resize(
        im_original, input_size, size_divisibility)
    return im, scale_w, scale_h, orig_w, orig_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run CharNet over a folder tree of document PNGs.")
    parser.add_argument("config_file", help="path to charnet config file")
    parser.add_argument("input_root",
                        help="root folder containing document subfolders with PNGs")
    parser.add_argument("output_root",
                        help="where to write results, preserving structure")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU workers for postprocessing (default: ncpus-1)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that already have JSON output")
    args = parser.parse_args(argv)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # ---- build model -------------------------------------------------------
    charnet = CharNet()
    charnet.load_state_dict(torch.load(cfg.WEIGHT, weights_only=True))
    charnet.eval()
    charnet.cuda()

    # ---- worker pool config ------------------------------------------------
    num_workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    num_workers = min(num_workers, 10)  # cap to available CPUs

    # Serialisable kwargs for OrientedTextPostProcessing (no cfg needed).
    postproc_kwargs = {
        "word_min_score": cfg.WORD_MIN_SCORE,
        "word_stride": cfg.WORD_STRIDE,
        "word_nms_iou_thresh": cfg.WORD_NMS_IOU_THRESH,
        "char_stride": cfg.CHAR_STRIDE,
        "char_min_score": cfg.CHAR_MIN_SCORE,
        "num_char_class": cfg.NUM_CHAR_CLASSES,
        "char_nms_iou_thresh": cfg.CHAR_NMS_IOU_THRESH,
        "char_dict_file": cfg.CHAR_DICT_FILE,
        "word_lexicon_path": cfg.WORD_LEXICON_PATH,
    }

    # ---- collect all images up-front ---------------------------------------
    image_tasks = []  # list of (im_file, image_id, out_dir)
    skipped = 0
    for doc in sorted(os.listdir(args.input_root)):
        doc_path = os.path.join(args.input_root, doc)
        if not os.path.isdir(doc_path):
            continue
        out_path = os.path.join(args.output_root, doc)
        for im_name in sorted(os.listdir(doc_path)):
            if not im_name.lower().endswith(".png"):
                continue
            image_id = os.path.splitext(im_name)[0]
            # Check if output already exists
            if args.skip_existing:
                json_path = os.path.join(out_path, f"{image_id}.json")
                if os.path.exists(json_path):
                    skipped += 1
                    continue
            image_tasks.append((
                os.path.join(doc_path, im_name),
                image_id,
                out_path,
            ))

    if skipped > 0:
        print(f"Skipped {skipped} images with existing outputs.")

    if not image_tasks:
        print("No images to process.")
        return

    print(f"Found {len(image_tasks)} images.  "
          f"Using 1 GPU + {num_workers} CPU workers.")

    # ---- pipelined processing ----------------------------------------------
    # Thread pool for prefetching the *next* image while the GPU is busy.
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    cpu_pool = ProcessPoolExecutor(max_workers=num_workers)
    futures = deque()
    MAX_PENDING = num_workers * 2  # back-pressure

    input_size = cfg.INPUT_SIZE
    size_div = cfg.SIZE_DIVISIBILITY
    char_dict_file = cfg.CHAR_DICT_FILE

    # Submit prefetch for the first image.
    next_prefetch = prefetch_pool.submit(
        _load_and_resize, image_tasks[0][0], input_size, size_div
    )

    with torch.no_grad(), cpu_pool:
        for i, (im_file, image_id, out_dir) in enumerate(image_tasks):
            # ---- wait for the prefetched image ----------------------------
            im, scale_w, scale_h, orig_w, orig_h = next_prefetch.result()

            # ---- kick off prefetch for the NEXT image ---------------------
            if i + 1 < len(image_tasks):
                next_prefetch = prefetch_pool.submit(
                    _load_and_resize, image_tasks[i + 1][0],
                    input_size, size_div,
                )

            # ---- GPU forward (fast) ---------------------------------------
            preds = charnet.forward_gpu(im)

            # ---- dispatch postprocessing + save to CPU pool ---------------
            os.makedirs(out_dir, exist_ok=True)
            futures.append(cpu_pool.submit(
                _postprocess_and_save,
                preds, scale_w, scale_h, orig_w, orig_h,
                image_id, out_dir, char_dict_file, postproc_kwargs,
            ))

            # ---- back-pressure: don't let pending futures grow unbounded --
            while len(futures) > MAX_PENDING:
                futures.popleft().result()

            if (i + 1) % 50 == 0 or i + 1 == len(image_tasks):
                print(f"  [{i+1}/{len(image_tasks)}] submitted")

        # Wait for all remaining tasks.
        while futures:
            futures.popleft().result()

    prefetch_pool.shutdown(wait=False)
    print("Done.")


if __name__ == '__main__':
    main()
