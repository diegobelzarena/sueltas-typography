import os
import time

import cv2
import numpy as np

import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.image_utils as image_utils
import craft_text_detector.torch_utils as torch_utils


def get_prediction(
    image,
    craft_net,
    refine_net=None,
    cuda: bool = False,
    ratio=None
):
    """
    Arguments:
        image: path to the image to be processed or numpy array or PIL image
        craft_net: craft net model
        refine_net: refine net model
        cuda: Use cuda for inference
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualizations of the detected characters/links,
         "times": elapsed times of the sub modules, in seconds}
    """
    # read/convert image
    image = image_utils.read_image(image)

    # preprocessing
    x = image_utils.normalizeMeanVariance(image)
    x = torch_utils.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = torch_utils.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch_utils.no_grad():
        y, feature = craft_net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link_w = y[0, :, :, 1].cpu().data.numpy()

    boxes_words, _ = craft_utils.getDetBoxes(score_text, score_link_w, 0.7, 0.4, 0.4, False)
    # coordinate adjustment
    boxes_words = craft_utils.adjustResultCoordinates(boxes_words, 1, 1)

    text_score_heatmap = image_utils.cvt2HeatmapImg(score_text)
    link_score_heatmap_w = image_utils.cvt2HeatmapImg(score_link_w)

    # refine link
    if refine_net is not None:
        with torch_utils.no_grad():
           y_refiner = refine_net(y, feature)
        score_link_l = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        boxes_lines, _ = craft_utils.getDetBoxes(score_text, score_link_l, 0.7, 0.4, 0.4, False)

        boxes_lines = craft_utils.adjustResultCoordinates(boxes_lines, 1, 1)
        
        link_score_heatmap_l = image_utils.cvt2HeatmapImg(score_link_l)

        return {
            "heatmaps": {
                "text_score_heatmap": text_score_heatmap,
                "link_score_heatmap_w": link_score_heatmap_w,
                "link_score_heatmap_l": link_score_heatmap_l,
            },
            "boxes_words": boxes_words,
            "boxes_lines": boxes_lines, 
        }
    else:
        return {
            "heatmaps": {
                "text_score_heatmap": text_score_heatmap,
                "link_score_heatmap_w": link_score_heatmap_w,
            },
            "boxes_words": boxes_words,
        }