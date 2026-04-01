"""
Custom OCR Predictor that extends DocTR's OCRPredictor with modifications.
Allows access to intermediate outputs and custom processing.
"""

import numpy as np
import torch
from torch import nn
from typing import Any, Union

from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.predictor.base import _OCRPredictor
from doctr.utils.geometry import detach_scores
from doctr.models._utils import get_language
from doctr.io.elements import Document
from .utils import gamma_correction

try:
    from .recognition import CustomRecognitionPredictor
    CUSTOM_RECO_AVAILABLE = True
except ImportError:
    CUSTOM_RECO_AVAILABLE = False
    CustomRecognitionPredictor = None


class CustomOCRPredictor(nn.Module, _OCRPredictor):
    """
    Custom OCR predictor with access to intermediate outputs.
    
    Extends DocTR's OCRPredictor but allows:
    - Access to detection probability maps
    - Access to cropped word images
    - Custom processing between detection and recognition
    - Modified document building
    """
    
    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: Union[RecognitionPredictor, 'CustomRecognitionPredictor'],
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        detect_language: bool = False,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        self.det_predictor = det_predictor.eval()
        self.reco_predictor = reco_predictor.eval()
        
        # Check if using custom recognition predictor
        self.using_custom_reco = CUSTOM_RECO_AVAILABLE and isinstance(reco_predictor, CustomRecognitionPredictor)
        
        _OCRPredictor.__init__(
            self,
            assume_straight_pages,
            straighten_pages,
            preserve_aspect_ratio,
            symmetric_pad,
            detect_orientation,
            **kwargs,
        )
        self.detect_orientation = detect_orientation
        self.detect_language = detect_language
        
        # Store intermediate outputs
        self.last_detection_maps = None
        self.last_crops = None
        self.last_loc_preds = None
        self.last_reco_logits = None
    
    @torch.inference_mode()
    def forward(
        self,
        pages: list[np.ndarray],
        return_intermediate: bool = False,
        det_kwargs: dict[str, Any] | None = None,
        reco_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Document | tuple[Document, dict]:
        """
        Forward pass with optional intermediate outputs.
        
        Args:
            pages: List of page images
            return_intermediate: If True, returns (Document, intermediate_dict)
            det_kwargs: Keyword arguments for detection predictor
            reco_kwargs: Keyword arguments for recognition predictor
            **kwargs: Additional arguments (backwards compatibility - passed to both if det_kwargs/reco_kwargs not specified)
            
        Returns:
            Document or (Document, intermediate_dict) if return_intermediate=True
        """
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        origin_page_shapes = [page.shape[:2] for page in pages]
        
        # Prepare kwargs for each predictor (backwards compatibility)
        det_params = det_kwargs if det_kwargs is not None else kwargs
        reco_params = reco_kwargs if reco_kwargs is not None else kwargs
        
        # Localize text elements
        loc_preds, out_maps = self.det_predictor(pages, return_maps=True, **det_params)
        
        # Store detection maps
        self.last_detection_maps = out_maps

        # Detect document rotation and rotate pages
        seg_maps = [
            np.where(out_map > getattr(self.det_predictor.model.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
            for out_map in out_maps
        ]
        
        if self.detect_orientation:
            general_pages_orientations, origin_pages_orientations = self._get_orientations(pages, seg_maps)
            orientations = [
                {"value": orientation_page, "confidence": None} for orientation_page in origin_pages_orientations
            ]
        else:
            orientations = None
            general_pages_orientations = None
            origin_pages_orientations = None
            
        if self.straighten_pages:
            pages = self._straighten_pages(pages, seg_maps, general_pages_orientations, origin_pages_orientations)
            origin_page_shapes = [page.shape[:2] for page in pages]
            loc_preds = self.det_predictor(pages, **det_params)

        assert all(len(loc_pred) == 1 for loc_pred in loc_preds), (
            "Detection Model in ocr_predictor should output only one class"
        )

        loc_preds = [list(loc_pred.values())[0] for loc_pred in loc_preds]
        # Detach objectness scores from loc_preds
        loc_preds, objectness_scores = detach_scores(loc_preds)

        
        # Store location predictions
        self.last_loc_preds = loc_preds

        # Apply hooks to loc_preds if any
        for hook in self.hooks:
            loc_preds = hook(loc_preds)

        # Crop images
        crops, loc_preds = self._prepare_crops(
            pages,
            loc_preds,
            assume_straight_pages=self.assume_straight_pages,
            assume_horizontal=self._page_orientation_disabled,
        )
        
        # Store crops
        self.last_crops = crops
        
        # Rectify crop orientation and get crop orientation predictions
        crop_orientations: Any = []
        if not self.assume_straight_pages:
            crops, loc_preds, _crop_orientations = self._rectify_crops(crops, loc_preds)
            crop_orientations = [
                {"value": orientation[0], "confidence": orientation[1]} for orientation in _crop_orientations
            ]

        # Identify character sequences
        reco_output = self.reco_predictor([crop for page_crops in crops for crop in page_crops], **reco_params)
        
        # Handle return_model_output case
        if isinstance(reco_output, dict):
            # When return_model_output=True, output is dict with 'preds' and 'out_map'
            word_preds = reco_output['preds']
            self.last_reco_logits = reco_output.get('out_map', None)
            batch_metadata = reco_output.get('metadata', None)
        else:
            word_preds = reco_output
            self.last_reco_logits = None
            batch_metadata = None
        
        if not crop_orientations:
            crop_orientations = [{"value": 0, "confidence": None} for _ in word_preds]

        boxes, text_preds, crop_orientations = self._process_predictions(loc_preds, word_preds, crop_orientations)

        if self.detect_language:
            languages = [get_language(" ".join([item[0] for item in text_pred])) for text_pred in text_preds]
            languages_dict = [{"value": lang[0], "confidence": lang[1]} for lang in languages]
        else:
            languages_dict = None

        # Build document
        # out = self.doc_builder(
        #     pages,
        #     boxes,
        #     objectness_scores,
        #     text_preds,
        #     origin_page_shapes,
        #     crop_orientations,
        #     orientations,
        #     languages_dict,
        # )
        geometries = self.get_word_geometries_from_detections(boxes, origin_page_shapes)
        out = {'words':
                [
                    {'text': text[0],
                    'confidence': text[1],
                    'geometry_normalized': geometry['geometry_normalized'],
                    'geometry_pixel': geometry['geometry_pixel'],
                    'center': geometry['center']
                    } 
                for text, geometry in zip(text_preds[0], geometries)]
        }

        logits_data = {
            'reco_logits': self.last_reco_logits,
            'batch_metadata': batch_metadata,
        }
        # if return_intermediate:
        #     intermediate = {
        #         'detection_maps': out_maps,
        #         'seg_maps': seg_maps,
        #         'crops': crops,
        #         'loc_preds': loc_preds,
        #         'boxes': boxes,
        #         'text_preds': text_preds,
        #         'crop_orientations': crop_orientations,
        #         'objectness_scores': objectness_scores,
        #         'reco_logits': self.last_reco_logits,  # Will be None if return_model_output not used
        #         'batch_metadata': batch_metadata,  # Metadata from recognition predictor if available
        #     }
        #     return out, intermediate
        
        return out, logits_data
    
    def run_detection_only(self, pages: list[np.ndarray], **kwargs) -> tuple[list, list]:
        """
        Run only detection, return location predictions and probability maps.
        
        Args:
            pages: List of page images
            **kwargs: Arguments for detection predictor
            
        Returns:
            (loc_preds, out_maps)
        """
        with torch.inference_mode():
            loc_preds, out_maps = self.det_predictor(pages, return_maps=True, **kwargs)
            
            assert all(len(loc_pred) == 1 for loc_pred in loc_preds)
            loc_preds = [list(loc_pred.values())[0] for loc_pred in loc_preds]
            loc_preds, _ = detach_scores(loc_preds)
            
            return loc_preds, out_maps
    
    def run_recognition_only(self, crops: list[np.ndarray], **kwargs) -> list | dict:
        """
        Run only recognition on pre-cropped images.
        
        Args:
            crops: List of cropped word images
            **kwargs: Arguments for recognition predictor
                     Use return_model_output=True to get logits/out_map
            
        Returns:
            List of recognition predictions, or dict with 'preds' and 'out_map' 
            if return_model_output=True
        """
        with torch.inference_mode():
            reco_output = self.reco_predictor(crops, **kwargs)
            
            # Store logits if available
            if isinstance(reco_output, dict):
                self.last_reco_logits = reco_output.get('out_map', None)
            else:
                self.last_reco_logits = None
            
            return reco_output
    
    def extract_crops_from_detections(
        self, 
        pages: list[np.ndarray], 
        loc_preds: list
    ) -> tuple[list, list]:
        """
        Extract word crops from pages given location predictions.
        
        Args:
            pages: List of page images
            loc_preds: Location predictions from detection
            
        Returns:
            (crops, updated_loc_preds)
        """
        crops, loc_preds = self._prepare_crops(
            pages,
            loc_preds,
            assume_straight_pages=self.assume_straight_pages,
            assume_horizontal=self._page_orientation_disabled,
        )
        
        if not self.assume_straight_pages:
            crops, loc_preds, _ = self._rectify_crops(crops, loc_preds)
        
        return crops, loc_preds
    
    def get_word_geometries_from_detections(
        self,
        loc_preds: list,
        page_shapes: list[tuple[int, int]]
    ) -> list[dict]:
        """
        Convert normalized location predictions to pixel coordinates.
        
        Args:
            loc_preds: Location predictions (normalized 0-1)
            page_shapes: Original page shapes (h, w)
            
        Returns:
            List of dicts with geometry info for each word
        """
        word_geometries = []
        
        for page_idx, page_loc_preds in enumerate(loc_preds):
            h, w = page_shapes[page_idx]
            
            for word_box in page_loc_preds:
                # word_box is typically shape (4, 2) with normalized coordinates
                geometry = word_box.tolist()  # Convert to list for JSON serialization
                
                # Calculate pixel coordinates
                pixel_coords = []
                for pt in word_box:
                    pixel_coords.append([int(pt[0] * w), int(pt[1] * h)])
                
                # Calculate bounding box
                pixel_array = np.array(pixel_coords)
                x1, y1 = pixel_array[:, 0].min(), pixel_array[:, 1].min()
                x2, y2 = pixel_array[:, 0].max(), pixel_array[:, 1].max()
                
                word_geometries.append({
                    'page': page_idx,
                    'geometry_normalized': geometry,
                    'geometry_pixel': pixel_coords,
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                })
        
        return word_geometries
