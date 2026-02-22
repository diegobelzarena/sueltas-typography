#!/usr/bin/env python
"""Run CharNet on a hierarchy of document images.

Each subdirectory of the input folder is treated as a document; all PNG
files within are processed.  The output folder mirrors the structure,
placing recognition results alongside the images.

Example:

    python scripts/run_charnet.py config.yaml input_root/ output_root/

This is a thin wrapper around the existing ``charnet`` machinery (see
``src/charnet/tools/test_net.py``).
"""

import os
import argparse
import json
import torch
import cv2
import numpy as np
from src.charnet.charnet.config import cfg
from src.charnet.charnet.modeling.model import CharNet


def save_char_probas(char_bboxes, char_scores, image_id, save_root):
    # same logic as charnet/tools/test_net.py
    from src.charnet.charnet.modeling.postprocessing import load_char_dict
    char_dict = load_char_dict(cfg.CHAR_DICT_FILE)
    char_ids = char_dict.keys()
    detections = []
    for char_bbox, char_score in zip(char_bboxes, char_scores):
        detection = {}
        l, r = int(min(char_bbox[:8:2])), int(max(char_bbox[:8:2]))
        t, b = int(min(char_bbox[1:8:2])), int(max(char_bbox[1:8:2]))
        detection['tblr'] = [t, b, l, r]
        sorted_ids = sorted(char_ids, key=lambda i: char_score[i], reverse=True)
        detection['labels'] = {char_dict[i].lower(): char_score[i].item()
                               for i in sorted_ids}
        detections.append(detection)
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, f"{image_id}.json"), "w") as f:
        json.dump(detections, f, indent=2)


def resize(im, size):
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height), interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def process_folder(charnet, folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for im_name in sorted(os.listdir(folder)):
        if not im_name.lower().endswith(".png"):
            continue
        im_file = os.path.join(folder, im_name)
        im_original = cv2.imread(im_file)
        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
        with torch.no_grad():
            char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
            save_word_recognition(
                word_instances,
                os.path.splitext(im_name)[0],
                out_folder,
                cfg.RESULTS_SEPARATOR,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run CharNet over a folder tree of document PNGs.")
    parser.add_argument("config_file", help="path to charnet config file")
    parser.add_argument("input_root", help="root folder containing document subfolders with PNGs")
    parser.add_argument("output_root", help="where to write results, preserving structure")
    args = parser.parse_args(argv)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    charnet = CharNet()
    charnet.load_state_dict(torch.load(cfg.WEIGHT))
    charnet.eval()
    charnet.cuda()

    for doc in sorted(os.listdir(args.input_root)):
        doc_path = os.path.join(args.input_root, doc)
        if not os.path.isdir(doc_path):
            continue
        out_path = os.path.join(args.output_root, doc)
        print(f"Processing document {doc}...")
        # new process folder that outputs JSONs
        os.makedirs(out_path, exist_ok=True)
        for im_name in sorted(os.listdir(doc_path)):
            if not im_name.lower().endswith(".png"):
                continue
            im_file = os.path.join(doc_path, im_name)
            im_original = cv2.imread(im_file)
            im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
            with torch.no_grad():
                char_bboxes, char_scores, word_instances = charnet(im, scale_w, scale_h, original_w, original_h)
                save_char_probas(
                    char_bboxes, char_scores,
                    os.path.splitext(im_name)[0], out_path
                )

if __name__ == '__main__':
    main()
