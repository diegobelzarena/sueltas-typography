from __future__ import absolute_import

import os
from typing import Optional

import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.file_utils as file_utils
import craft_text_detector.image_utils as image_utils
import craft_text_detector.predict as predict
import craft_text_detector.torch_utils as torch_utils

__version__ = "0.4.3"


__all__ = [
    "read_image",
    "load_craftnet_model",
    "load_refinenet_model",
    "get_prediction",
    "export_detected_regions",
    "export_extra_results",
    "empty_cuda_cache",
    "Craft",
]

read_image = image_utils.read_image
load_craftnet_model = craft_utils.load_craftnet_model
load_refinenet_model = craft_utils.load_refinenet_model
get_prediction = predict.get_prediction
export_detected_regions = file_utils.export_detected_regions
export_extra_results = file_utils.export_extra_results
empty_cuda_cache = torch_utils.empty_cuda_cache


class Craft:
    def __init__(
        self,
        cuda=False,
        refiner=True,
        weight_path_craft_net: Optional[str] = None,
        weight_path_refine_net: Optional[str] = None,
    ):
        """
        Arguments:
            cuda: Use cuda for inference
            refiner: enable link refiner
        """
        self.craft_net = None
        self.refine_net = None
        self.cuda = cuda
        self.refiner = refiner

        # load craftnet
        self.load_craftnet_model(weight_path_craft_net)
        # load refinernet if required
        if refiner:
           self.load_refinenet_model(weight_path_refine_net)

    def load_craftnet_model(self, weight_path: Optional[str] = None):
        """
        Loads craftnet model
        """
        self.craft_net = load_craftnet_model(self.cuda, weight_path=weight_path)

    def load_refinenet_model(self, weight_path: Optional[str] = None):
        """
        Loads refinenet model
        """
        self.refine_net = load_refinenet_model(self.cuda, weight_path=weight_path)

    def unload_craftnet_model(self):
        """
        Unloads craftnet model
        """
        self.craft_net = None
        empty_cuda_cache()

    def unload_refinenet_model(self):
        """
        Unloads refinenet model
        """
        self.refine_net = None
        empty_cuda_cache()

    def detect_text(self, image, image_path=None, ratio=None):
        """
        Arguments:
            image: path to the image to be processed or numpy array or PIL image

        Output:
            {
                "masks": lists of predicted masks 2d as bool array,
                "boxes": list of coords of points of predicted boxes,
                "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
                "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
                "heatmaps": visualization of the detected characters/links,
                "text_crop_paths": list of paths of the exported text boxes/polys,
                "times": elapsed times of the sub modules, in seconds
            }
        """

        if image_path is not None:
            print("Argument 'image_path' is deprecated, use 'image' instead.")
            image = image_path

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            cuda=self.cuda,
            ratio=ratio
        )
        return prediction_result
