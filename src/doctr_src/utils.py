"""Utility functions for the DocTR character-extraction path.

Provides CTC-logit decoding, per-word logit extraction, word-level
segmentation using logit-initialised boxes, and gamma correction.
"""

from __future__ import annotations

import cv2
import numpy as np

from image_processing.char_segment import char_segment_logit_init


# ---------------------------------------------------------------------------
# Gamma correction (ported from extract_chars/modules/image_processing.py)
# ---------------------------------------------------------------------------

def gamma_correction(img: np.ndarray) -> np.ndarray:
    """Apply adaptive gamma correction until the 10 % quantile < 0.5.

    Args:
        img: Grayscale image, values in [0, 1].

    Returns:
        Gamma-corrected image.
    """
    gammas = [1.5, 2.0, 2.5, 3.0]
    cond = np.quantile(img, 0.1) < 0.5
    gamma = 1.0
    while not cond and gammas:
        gamma = gammas.pop(0)
        img = img ** gamma
        cond = np.quantile(img, 0.1) < 0.5
    print(f"  Applied gamma correction with gamma={gamma}")
    return img


# ---------------------------------------------------------------------------
# CRNN logit → character positions
# ---------------------------------------------------------------------------

def rcnn_positions(
    logits_word: np.ndarray,
    word_img_shape: tuple[int, int],
    resize_word: tuple[float, float],
    pad_word: tuple[int, int],
    delta_y1: int,
    delta_y2: int,
    delta_x1: int,
    delta_x2: int,
) -> list[int]:
    """Decode CTC-aligned CRNN logits into pixel-level character boundaries.

    Args:
        logits_word: CRNN output logits, shape ``(seq_len, num_classes)``.
        word_img_shape: ``(height, width)`` of the word crop.
        resize_word: ``(scale_h, scale_w)`` from preprocessing.
        pad_word: ``(pad_left, pad_right)`` from preprocessing.
        delta_y1: Top padding delta (typically ≤ 0).
        delta_y2: Bottom padding delta.
        delta_x1: Left padding delta (typically ≤ 0).
        delta_x2: Right padding delta.

    Returns:
        List of pixel x-positions delineating character boundaries
        (length = num_chars + 1, ending at the right edge of the word).
    """
    seq_len = logits_word.shape[0]
    max_logits = np.argmax(logits_word, axis=1)
    pos_ctc: list[int] = []
    prev_char = None
    for i_c, char in enumerate(max_logits):
        if char != 126 and char != prev_char:
            pos = int(
                (i_c * ((word_img_shape[1] - (abs(delta_x1) + delta_x2))
                        * resize_word[1] + pad_word[1] + pad_word[0]))
                / (seq_len * resize_word[1])
            ) + abs(delta_x1)
            pos_ctc.append(pos)
        prev_char = char
    pos_ctc.append(word_img_shape[1] - delta_x2 - 1)
    return pos_ctc


# ---------------------------------------------------------------------------
# Per-word logit extraction
# ---------------------------------------------------------------------------

def get_word_logits(
    word_height: int,
    logits: list[np.ndarray],
    o_idxs: np.ndarray,
    r_scls: np.ndarray,
    pads: np.ndarray,
    b_idxs: np.ndarray,
    indx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract CRNN logits and preprocessing metadata for a single word.

    Returns:
        ``(logits_word, resize_word, pad_word, pad_w)``
    """
    b_idx = b_idxs[o_idxs == indx][0]
    logits_word = logits[b_idx][o_idxs[b_idxs == b_idx] == indx][0]
    resize_word = r_scls[o_idxs == indx][0]
    pad_word = pads[o_idxs == indx][0]
    pad_w = int(word_height / 5)
    return logits_word, resize_word, pad_word, pad_w


# ---------------------------------------------------------------------------
# Word-level segmentation using logit-initialised boxes
# ---------------------------------------------------------------------------

def char_segment_word(
    word_img: np.ndarray,
    logits_word: np.ndarray,
    resize_word: tuple[float, float],
    pad_word: tuple[int, int],
    delta_y1: int,
    delta_y2: int,
    delta_x1: int,
    delta_x2: int,
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Segment a single word image into character masks using CRNN logits.

    1. Decodes CTC logits into character boundary positions.
    2. Builds initial character boxes from those positions.
    3. Applies contrast normalisation to the word crop.
    4. Runs :func:`char_segment_logit_init` for min-cost-path refinement.

    Returns:
        List of ``((t, b, l, r), mask)`` tuples, one per detected character.
    """
    pos_ctc = rcnn_positions(
        logits_word, word_img.shape, resize_word, pad_word,
        delta_y1, delta_y2, delta_x1, delta_x2,
    )

    top_pad = abs(delta_y1)
    bottom_pad = delta_y2

    tblrs = np.array([
        [top_pad, word_img.shape[0] - bottom_pad,
         max(0, min(pos_ctc[j], word_img.shape[1] - 1)),
         max(0, min(pos_ctc[j + 1] - 1, word_img.shape[1] - 1))]
        for j in range(len(pos_ctc) - 1)
    ])

    if len(tblrs) == 0:
        return []

    # Filter out invalid boxes
    valid_mask = (tblrs[:, 1] > tblrs[:, 0]) & (tblrs[:, 3] > tblrs[:, 2])
    tblrs = tblrs[valid_mask]
    if len(tblrs) == 0:
        return []

    widths = tblrs[:, 3] - tblrs[:, 2]
    refwidth = float(np.mean(widths)) if np.any(widths) else 1.0
    box_clu = np.ones(len(tblrs), dtype=int)

    # Contrast normalisation
    kh = (word_img.shape[0] // 2) * 2 + 1
    kw = (word_img.shape[1] // 2) * 2 + 1
    blur_img = cv2.GaussianBlur(
        word_img, (kh, kw),
        sigmaX=word_img.shape[0] // 2,
        sigmaY=word_img.shape[1] // 2,
    )
    cont_img = word_img / (blur_img + 1e-6)
    pre_img = cont_img.copy()
    pre_img -= pre_img.min()
    pre_img /= pre_img.max()

    return char_segment_logit_init(
        pre_img, tblrs.copy(), box_clu, refwidth,
        top_pad, bottom_pad, widths,
    )
