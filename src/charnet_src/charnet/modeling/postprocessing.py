# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch import nn
import numpy as np
import cv2
import editdistance
from .utils import rotate_rect, rotate_rect_vectorized
from .rotated_nms import nms, nms_with_char_cls, \
    softnms, nms_poly
from shapely.geometry import Polygon
import pyclipper


# -----------------------------------------------------------------------
# BK-tree for fast approximate string matching against the lexicon.
# -----------------------------------------------------------------------

class _BKNode:
    __slots__ = ("word", "children")

    def __init__(self, word):
        self.word = word
        self.children = {}


class BKTree:
    """A BK-tree for edit-distance queries.  Build once from the lexicon,
    then call ``search(word, max_dist)`` to find the closest match in
    roughly *O(log n)* time instead of scanning all entries."""

    def __init__(self, words):
        it = iter(words)
        self.root = _BKNode(next(it))
        for w in it:
            self._insert(self.root, w)

    @staticmethod
    def _insert(node, word):
        d = editdistance.eval(node.word, word)
        while d in node.children:
            node = node.children[d]
            d = editdistance.eval(node.word, word)
        node.children[d] = _BKNode(word)

    def search(self, word, max_dist):
        """Return ``(best_distance, best_word)`` within *max_dist*, or
        ``(None, None)`` if nothing is close enough."""
        best_dist = max_dist + 1
        best_word = None
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = editdistance.eval(word, node.word)
            if d < best_dist:
                best_dist = d
                best_word = node.word
                if d == 0:
                    return 0, best_word
            # Only explore children within the triangle-inequality window.
            lo = max(0, d - best_dist + 1)
            hi = d + best_dist
            for dist_key, child in node.children.items():
                if lo <= dist_key <= hi:
                    stack.append(child)
        if best_dist <= max_dist:
            return best_dist, best_word
        return None, None


def load_lexicon(path):
    lexicon = list()
    with open(path, 'rt') as fr:
        for line in fr:
            if line.startswith('#'):
                pass
            else:
                lexicon.append(line.strip())
    return lexicon


def load_char_dict(path, seperator=chr(31)):
    char_dict = dict()
    with open(path, 'rt') as fr:
        for line in fr:
            sp = line.strip('\n').split(seperator)
            char_dict[int(sp[1])] = sp[0].upper()
    return char_dict


class WordInstance:
    def __init__(self, word_bbox, word_bbox_score, text, text_score,
                 char_scores, char_bboxes):
        self.word_bbox = word_bbox
        self.word_bbox_score = word_bbox_score
        self.text = text
        self.text_score = text_score
        self.char_scores = char_scores
        self.char_bboxes = char_bboxes


class OrientedTextPostProcessing(nn.Module):
    def __init__(
            self, word_min_score, word_stride,
            word_nms_iou_thresh, char_stride,
            char_min_score, num_char_class,
            char_nms_iou_thresh, char_dict_file,
            word_lexicon_path
    ):
        super(OrientedTextPostProcessing, self).__init__()
        self.word_min_score = word_min_score
        self.word_stride = word_stride
        self.word_nms_iou_thresh = word_nms_iou_thresh
        self.char_stride = char_stride
        self.char_min_score = char_min_score
        self.num_char_class = num_char_class
        self.char_nms_iou_thresh = char_nms_iou_thresh
        self.char_dict = load_char_dict(char_dict_file)
        self.lexicon = load_lexicon(word_lexicon_path)
        # Build a BK-tree over upper-cased lexicon for ~O(log n) lookups
        # instead of scanning all 87 k entries per uncertain word.
        self._lexicon_upper = [w.upper() for w in self.lexicon]
        self.bk_tree = BKTree(self._lexicon_upper)

    def forward(
            self, pred_word_fg, pred_word_tblr,
            pred_word_orient, pred_char_fg,
            pred_char_tblr, pred_char_cls,
            im_scale_w, im_scale_h,
            original_im_w, original_im_h
    ):
        ss_word_bboxes = self.parse_word_bboxes(
            pred_word_fg, pred_word_tblr, pred_word_orient,
            im_scale_w, im_scale_h, original_im_w, original_im_h
        )
        char_bboxes, char_scores = self.parse_char(
            pred_word_fg, pred_char_fg, pred_char_tblr, pred_char_cls,
            im_scale_w, im_scale_h, original_im_w, original_im_h
        )
        word_instances = self.parse_words(
            ss_word_bboxes, char_bboxes,
            char_scores, self.char_dict
        )

        word_instances = self.filter_word_instances(word_instances, self.lexicon)

        return char_bboxes, char_scores, word_instances

    def parse_word_bboxes(
            self, pred_word_fg, pred_word_tblr,
            pred_word_orient, scale_w, scale_h,
            W, H
    ):
        word_stride = self.word_stride
        word_keep_rows, word_keep_cols = np.where(pred_word_fg > self.word_min_score)
        n = word_keep_rows.shape[0]
        if n == 0:
            return np.zeros((0, 9), dtype=np.float32)

        y = word_keep_rows.astype(np.float32)
        x = word_keep_cols.astype(np.float32)
        tblr = pred_word_tblr[:, word_keep_rows, word_keep_cols]  # (4, n)
        t, b, l, r = tblr[0], tblr[1], tblr[2], tblr[3]
        orient = pred_word_orient[word_keep_rows, word_keep_cols]
        scores = pred_word_fg[word_keep_rows, word_keep_cols]

        sw = np.float32(scale_w * word_stride)
        sh = np.float32(scale_h * word_stride)

        x1 = sw * (x - l)
        y1 = sh * (y - t)
        x2 = sw * (x + r)
        y2 = sh * (y + b)
        cx = sw * x
        cy = sh * y

        oriented_word_bboxes = np.empty((n, 9), dtype=np.float32)
        oriented_word_bboxes[:, :8] = rotate_rect_vectorized(x1, y1, x2, y2, orient, cx, cy)
        oriented_word_bboxes[:, 8] = scores

        keep, oriented_word_bboxes = nms(oriented_word_bboxes, self.word_nms_iou_thresh, num_neig=1)
        oriented_word_bboxes = oriented_word_bboxes[keep]
        oriented_word_bboxes[:, :8] = oriented_word_bboxes[:, :8].round()
        oriented_word_bboxes[:, 0:8:2] = np.maximum(0, np.minimum(W-1, oriented_word_bboxes[:, 0:8:2]))
        oriented_word_bboxes[:, 1:8:2] = np.maximum(0, np.minimum(H-1, oriented_word_bboxes[:, 1:8:2]))
        return oriented_word_bboxes

    def parse_char(
            self, pred_word_fg, pred_char_fg,
            pred_char_tblr, pred_char_cls,
            scale_w, scale_h, W, H
    ):
        char_stride = self.char_stride
        if pred_word_fg.shape == pred_char_fg.shape:
            char_keep_rows, char_keep_cols = np.where(
                (pred_word_fg > self.word_min_score) & (pred_char_fg > self.char_min_score))
        else:
            th, tw = pred_char_fg.shape
            word_fg_mask = cv2.resize((pred_word_fg > self.word_min_score).astype(np.uint8),
                                      (tw, th), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            char_keep_rows, char_keep_cols = np.where(
                word_fg_mask & (pred_char_fg > self.char_min_score))

        n = char_keep_rows.shape[0]
        if n == 0:
            return np.zeros((0, 9), dtype=np.float32), np.zeros((0, self.num_char_class), dtype=np.float32)

        y = char_keep_rows.astype(np.float32)
        x = char_keep_cols.astype(np.float32)
        tblr = pred_char_tblr[:, char_keep_rows, char_keep_cols]  # (4, n)
        t, b, l, r = tblr[0], tblr[1], tblr[2], tblr[3]
        orient = np.zeros(n, dtype=np.float32)  # char orient is always 0
        scores = pred_char_fg[char_keep_rows, char_keep_cols]

        sw = np.float32(scale_w * char_stride)
        sh = np.float32(scale_h * char_stride)

        x1 = sw * (x - l)
        y1 = sh * (y - t)
        x2 = sw * (x + r)
        y2 = sh * (y + b)
        cx = sw * x
        cy = sh * y

        oriented_char_bboxes = np.empty((n, 9), dtype=np.float32)
        oriented_char_bboxes[:, :8] = rotate_rect_vectorized(x1, y1, x2, y2, orient, cx, cy)
        oriented_char_bboxes[:, 8] = scores
        char_scores = pred_char_cls[:, char_keep_rows, char_keep_cols].T.astype(np.float32)  # (n, C)

        keep, oriented_char_bboxes, char_scores = nms_with_char_cls(
            oriented_char_bboxes, char_scores, self.char_nms_iou_thresh, num_neig=1
        )
        oriented_char_bboxes = oriented_char_bboxes[keep]
        oriented_char_bboxes[:, :8] = oriented_char_bboxes[:, :8].round()
        oriented_char_bboxes[:, 0:8:2] = np.maximum(0, np.minimum(W-1, oriented_char_bboxes[:, 0:8:2]))
        oriented_char_bboxes[:, 1:8:2] = np.maximum(0, np.minimum(H-1, oriented_char_bboxes[:, 1:8:2]))
        char_scores = char_scores[keep]
        return oriented_char_bboxes, char_scores

    def filter_word_instances(self, word_instances, lexicon):
        bk = self.bk_tree

        def filter_and_correct(word_ins):
            if len(word_ins.text) < 3:
                return None
            elif word_ins.text.isalpha():
                if word_ins.text_score >= 0.80:
                    if word_ins.text_score >= 0.98:
                        return word_ins
                    else:
                        # BK-tree: find closest match within edit distance 1
                        dist, voc = bk.search(word_ins.text.upper(), max_dist=1)
                        if dist is not None:
                            word_ins.text = voc
                            word_ins.text_edst = dist
                            return word_ins
                        else:
                            return None
                else:
                    return None
            else:
                if word_ins.text_score >= 0.90:
                    return word_ins
                else:
                    return None

        valid_word_instances = list()
        for word_ins in word_instances:
            word_ins = filter_and_correct(word_ins)
            if word_ins is not None:
                valid_word_instances.append(word_ins)
        return valid_word_instances

    def nms_word_instances(self, word_instances, h, w, edst=False):
        word_bboxes = np.zeros((len(word_instances), 9), dtype=np.float32)
        for idx, word_ins in enumerate(word_instances):
            word_bboxes[idx, :8] = word_ins.word_bbox
            word_bboxes[idx, 8] = word_ins.word_bbox_score * 1 + word_ins.text_score
            if edst is True:
                text_edst = getattr(word_ins, 'text_edst', 0)
                word_bboxes[idx, 8] -= (word_ins.text_score / len(word_ins.text)) * text_edst
        keep, word_bboxes = nms(word_bboxes, self.word_nms_iou_thresh, num_neig=0)
        word_bboxes = word_bboxes[keep]
        word_bboxes[:, :8] = word_bboxes[:, :8].round()
        word_bboxes[:, 0:8:2] = np.maximum(0, np.minimum(w-1, word_bboxes[:, 0:8:2]))
        word_bboxes[:, 1:8:2] = np.maximum(0, np.minimum(h-1, word_bboxes[:, 1:8:2]))
        word_instances = [word_instances[idx] for idx in keep]
        for word_ins, word_bbox, in zip(word_instances, word_bboxes):
            word_ins.word_bbox[:8] = word_bbox[:8]
        return word_instances

    def parse_words(self, word_bboxes, char_bboxes, char_scores, char_dict):
        def decode(char_scores):
            max_indices = char_scores.argmax(axis=1)
            text = [char_dict[idx] for idx in max_indices]
            scores = [char_scores[idx, max_indices[idx]] for idx in range(max_indices.shape[0])]
            return ''.join(text), np.array(scores, dtype=np.float32).mean()

        def recog(word_bbox, char_bboxes, char_scores):
            word_vec = np.array([1, 0], dtype=np.float32)
            char_vecs = (char_bboxes.reshape((-1, 4, 2)) - word_bbox[0:2]).mean(axis=1)
            proj = char_vecs.dot(word_vec)
            order = np.argsort(proj)
            text, score = decode(char_scores[order])
            return text, score, char_scores[order], char_bboxes[order]

        word_bbox_scores = word_bboxes[:, 8]
        char_bbox_scores = char_bboxes[:, 8]
        word_bboxes = word_bboxes[:, :8]
        char_bboxes = char_bboxes[:, :8]

        num_word = word_bboxes.shape[0]
        num_char = char_bboxes.shape[0]

        if num_word == 0 or num_char == 0:
            return []

        # ------------------------------------------------------------------
        # AABB pre-filter: compute axis-aligned bounding boxes and use
        # vectorised comparisons to rule out pairs that cannot possibly
        # overlap.  Only the surviving (char, word) pairs are tested with
        # the expensive Shapely polygon intersection.
        # ------------------------------------------------------------------
        word_xs = word_bboxes[:, 0:8:2]  # (W, 4)
        word_ys = word_bboxes[:, 1:8:2]
        w_xmin = word_xs.min(axis=1)     # (W,)
        w_xmax = word_xs.max(axis=1)
        w_ymin = word_ys.min(axis=1)
        w_ymax = word_ys.max(axis=1)

        char_xs = char_bboxes[:, 0:8:2]  # (C, 4)
        char_ys = char_bboxes[:, 1:8:2]
        c_xmin = char_xs.min(axis=1)     # (C,)
        c_xmax = char_xs.max(axis=1)
        c_ymin = char_ys.min(axis=1)
        c_ymax = char_ys.max(axis=1)

        # Broadcast: (C, 1) vs (W,) → (C, W) boolean masks
        no_overlap = (
            (c_xmin[:, None] > w_xmax[None, :]) |
            (c_xmax[:, None] < w_xmin[None, :]) |
            (c_ymin[:, None] > w_ymax[None, :]) |
            (c_ymax[:, None] < w_ymin[None, :])
        )
        # candidate_pairs: list of (char_idx, word_idx) that pass the AABB test
        maybe_overlap = ~no_overlap  # (C, W)

        # Pre-build Shapely polygons only for items that have at least one candidate
        word_polys = [None] * num_word
        char_polys = [None] * num_char

        def _get_word_poly(j):
            if word_polys[j] is None:
                b = word_bboxes[j]
                word_polys[j] = Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])])
            return word_polys[j]

        def _get_char_poly(i):
            if char_polys[i] is None:
                b = char_bboxes[i]
                char_polys[i] = Polygon([(b[0], b[1]), (b[2], b[3]), (b[4], b[5]), (b[6], b[7])])
            return char_polys[i]

        word_chars = [list() for _ in range(num_word)]

        for idx in range(num_char):
            # Candidate word indices from AABB filter
            cand_words = np.where(maybe_overlap[idx])[0]
            if cand_words.size == 0:
                continue

            char_poly = _get_char_poly(idx)
            char_area = char_poly.area
            if char_area == 0:
                continue

            best_iou = 0.0
            best_jdx = -1
            for jdx in cand_words:
                word_poly = _get_word_poly(jdx)
                inter = char_poly.intersection(word_poly).area
                iou = inter / (char_area + word_poly.area - inter) if (char_area + word_poly.area - inter) > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_jdx = jdx
            if best_iou > 0:
                word_chars[best_jdx].append(idx)

        word_instances = list()
        for idx in range(num_word):
            char_indices = word_chars[idx]
            if len(char_indices) > 0:
                text, text_score, tmp_char_scores, tmp_char_bboxes = recog(
                    word_bboxes[idx],
                    char_bboxes[char_indices],
                    char_scores[char_indices]
                )
                word_instances.append(WordInstance(
                    word_bboxes[idx],
                    word_bbox_scores[idx],
                    text, text_score,
                    tmp_char_scores,
                    tmp_char_bboxes,
                ))
        return word_instances
