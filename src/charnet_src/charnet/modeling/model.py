# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from charnet.modeling.backbone.resnet import resnet50
from charnet.modeling.backbone.hourglass import hourglass88
from charnet.modeling.backbone.decoder import Decoder
from collections import OrderedDict
from torch.functional import F
from charnet.modeling.layers import Scale
from .postprocessing import OrientedTextPostProcessing
from charnet.config import cfg


def _conv3x3_bn_relu(in_channels, out_channels, dilation=1):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False
        )),
        ("bn", nn.BatchNorm2d(out_channels)),
        ("relu", nn.ReLU())
    ]))


def to_numpy_or_none(*tensors):
    results = []
    for t in tensors:
        if t is None:
            results.append(None)
        else:
            results.append(t.cpu().numpy())
    return results


class WordDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, dilation=1):
        super(WordDetector, self).__init__()
        self.word_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels, dilation
        )
        self.word_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels, dilation
        )
        self.word_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.word_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)
        self.orient_pred = nn.Conv2d(bottleneck_channels, 1, kernel_size=1)

    def forward(self, x):
        feat = self.word_det_conv_final(x)

        pred_word_fg = self.word_fg_pred(self.word_fg_feat(feat))

        word_regression_feat = self.word_regression_feat(feat)
        pred_word_tblr = F.relu(self.word_tblr_pred(word_regression_feat)) * 10.
        pred_word_orient = self.orient_pred(word_regression_feat)

        return pred_word_fg, pred_word_tblr, pred_word_orient


class CharDetector(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, curved_text_on=False):
        super(CharDetector, self).__init__()
        self.character_det_conv_final = _conv3x3_bn_relu(
            in_channels, bottleneck_channels
        )
        self.char_fg_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_regression_feat = _conv3x3_bn_relu(
            bottleneck_channels, bottleneck_channels
        )
        self.char_fg_pred = nn.Conv2d(bottleneck_channels, 2, kernel_size=1)
        self.char_tblr_pred = nn.Conv2d(bottleneck_channels, 4, kernel_size=1)

    def forward(self, x):
        feat = self.character_det_conv_final(x)

        pred_char_fg = self.char_fg_pred(self.char_fg_feat(feat))
        char_regression_feat = self.char_regression_feat(feat)
        pred_char_tblr = F.relu(self.char_tblr_pred(char_regression_feat)) * 10.
        pred_char_orient = None

        return pred_char_fg, pred_char_tblr, pred_char_orient


class CharRecognizer(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, num_classes):
        super(CharRecognizer, self).__init__()

        self.body = nn.Sequential(
            _conv3x3_bn_relu(in_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
            _conv3x3_bn_relu(bottleneck_channels, bottleneck_channels),
        )
        self.classifier = nn.Conv2d(bottleneck_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        feat = self.body(feat)
        return self.classifier(feat)


class CharNet(nn.Module):
    def __init__(self, backbone=hourglass88()):
        super(CharNet, self).__init__()
        self.backbone = backbone
        decoder_channels = 256
        bottleneck_channels = 128
        self.word_detector = WordDetector(
            decoder_channels, bottleneck_channels,
            dilation=cfg.WORD_DETECTOR_DILATION
        )
        self.char_detector = CharDetector(
            decoder_channels,
            bottleneck_channels
        )
        self.char_recognizer = CharRecognizer(
            decoder_channels, bottleneck_channels,
            num_classes=cfg.NUM_CHAR_CLASSES
        )

        args = {
            "word_min_score": cfg.WORD_MIN_SCORE,
            "word_stride": cfg.WORD_STRIDE,
            "word_nms_iou_thresh": cfg.WORD_NMS_IOU_THRESH,
            "char_stride": cfg.CHAR_STRIDE,
            "char_min_score": cfg.CHAR_MIN_SCORE,
            "num_char_class": cfg.NUM_CHAR_CLASSES,
            "char_nms_iou_thresh": cfg.CHAR_NMS_IOU_THRESH,
            "char_dict_file": cfg.CHAR_DICT_FILE,
            "word_lexicon_path": cfg.WORD_LEXICON_PATH
        }

        self.post_processing = OrientedTextPostProcessing(**args)

        self.transform = self.build_transform()

    def forward(self, im, im_scale_w, im_scale_h, original_im_w, original_im_h):
        im = self.transform(im).cuda()
        im = im.unsqueeze(0)
        features = self.backbone(im)

        pred_word_fg, pred_word_tblr, pred_word_orient = self.word_detector(features)
        pred_char_fg, pred_char_tblr, pred_char_orient = self.char_detector(features)
        recognition_results = self.char_recognizer(features)

        pred_word_fg = F.softmax(pred_word_fg, dim=1)
        pred_char_fg = F.softmax(pred_char_fg, dim=1)
        pred_char_cls = F.softmax(recognition_results, dim=1)

        pred_word_fg, pred_word_tblr, \
        pred_word_orient, pred_char_fg, \
        pred_char_tblr, pred_char_cls, \
        pred_char_orient = to_numpy_or_none(
            pred_word_fg, pred_word_tblr,
            pred_word_orient, pred_char_fg,
            pred_char_tblr, pred_char_cls,
            pred_char_orient
        )

        char_bboxes, char_scores, word_instances = self.post_processing(
            pred_word_fg[0, 1], pred_word_tblr[0],
            pred_word_orient[0, 0], pred_char_fg[0, 1],
            pred_char_tblr[0], pred_char_cls[0],
            im_scale_w, im_scale_h,
            original_im_w, original_im_h
        )

        return char_bboxes, char_scores, word_instances

    def forward_gpu(self, im):
        """Run only the GPU part (backbone + heads + softmax).

        Parameters
        ----------
        im : numpy array (H, W, 3) BGR uint8 — already resized.

        Returns
        -------
        dict of numpy arrays ready for CPU post-processing.
        """
        im_t = self.transform(im).cuda()
        im_t = im_t.unsqueeze(0)

        # Use automatic mixed precision (FP16) to roughly halve GPU
        # memory bandwidth and increase throughput on modern GPUs.
        with torch.cuda.amp.autocast(dtype=torch.float16):
            features = self.backbone(im_t)
            pred_word_fg, pred_word_tblr, pred_word_orient = self.word_detector(features)
            pred_char_fg, pred_char_tblr, pred_char_orient = self.char_detector(features)
            recognition_results = self.char_recognizer(features)

        # Softmax in FP32 for numerical stability.
        pred_word_fg = F.softmax(pred_word_fg.float(), dim=1)
        pred_char_fg = F.softmax(pred_char_fg.float(), dim=1)
        pred_char_cls = F.softmax(recognition_results.float(), dim=1)

        pred_word_fg, pred_word_tblr, \
        pred_word_orient, pred_char_fg, \
        pred_char_tblr, pred_char_cls, \
        pred_char_orient = to_numpy_or_none(
            pred_word_fg, pred_word_tblr,
            pred_word_orient, pred_char_fg,
            pred_char_tblr, pred_char_cls,
            pred_char_orient
        )

        return {
            "pred_word_fg": pred_word_fg[0, 1],
            "pred_word_tblr": pred_word_tblr[0],
            "pred_word_orient": pred_word_orient[0, 0],
            "pred_char_fg": pred_char_fg[0, 1],
            "pred_char_tblr": pred_char_tblr[0],
            "pred_char_cls": pred_char_cls[0],
        }

    def postprocess(self, preds, im_scale_w, im_scale_h, original_im_w, original_im_h):
        """Run only the CPU post-processing on pre-computed predictions."""
        return self.post_processing(
            preds["pred_word_fg"], preds["pred_word_tblr"],
            preds["pred_word_orient"], preds["pred_char_fg"],
            preds["pred_char_tblr"], preds["pred_char_cls"],
            im_scale_w, im_scale_h,
            original_im_w, original_im_h,
        )

    def build_transform(self):
        """Build an image transform that converts a BGR uint8 numpy array
        to a normalised RGB float tensor — without the unnecessary
        numpy → PIL → tensor round-trip of the original code.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        def _transform(im_bgr_uint8):
            # (H, W, 3) uint8 BGR → (3, H, W) float32 RGB, normalised
            im = im_bgr_uint8[:, :, ::-1].copy()          # BGR → RGB (contiguous)
            im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            im = (im - mean) / std
            return im

        return _transform
