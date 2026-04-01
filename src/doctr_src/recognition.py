"""
Custom Recognition Predictor that extends DocTR's RecognitionPredictor.
Allows modifications to recognition pipeline while maintaining compatibility.
"""

from collections.abc import Sequence
from typing import Any
import math

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

from doctr.transforms import Resize
from doctr.utils.multithreading import multithread_exec
from doctr.models.preprocessor import PreProcessor
from doctr.models import recognition
from doctr.models.utils import _CompiledModule
from doctr.models.utils import set_device_and_dtype
from doctr.models.recognition.predictor._utils import remap_preds, split_crops


class CustomPreProcessor(nn.Module):
    """
    Custom preprocessor with modifiable behavior.
    
    Implements an abstract preprocessor object which performs casting, resizing, 
    batching and normalization. This is a copy of DocTR's PreProcessor that can be modified.

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        dynamic_width_batching: if True, batch by aspect ratio with dynamic padding
        **kwargs: additional arguments for the resizing operation
    """

    def __init__(
        self,
        output_size: tuple[int, int],
        batch_size: int,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        dynamic_width_batching: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize: T.Resize = Resize(output_size, **kwargs)
        # Perform the division by 255 at the same time
        self.normalize = T.Normalize(mean, std)
        self.dynamic_width_batching = dynamic_width_batching
        self.output_height = output_size[0]
        self.output_width = output_size[1]
        # Store metadata from last preprocessing
        self.last_metadata = None

    def batch_inputs(self, samples: list[torch.Tensor]) -> list[torch.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
            samples: list of samples of shape (C, H, W)

        Returns:
            list of batched samples (*, C, H, W)
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            torch.stack(samples[idx * self.batch_size : min((idx + 1) * self.batch_size, len(samples))], dim=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def batch_inputs_dynamic(
        self, 
        samples: list[torch.Tensor], 
        original_shapes: list[tuple[int, int]]
    ) -> tuple[list[torch.Tensor], dict]:
        """
        Batch samples by aspect ratio with dynamic width padding.
        
        Args:
            samples: list of resized samples of shape (C, H, W) 
            original_shapes: list of original (H, W) before resizing
            
        Returns:
            batches: list of batched samples with dynamic width
            metadata: dict containing indices, scales, padding info per batch
        """
        # Calculate aspect ratios and sort
        aspect_ratios = [s.shape[2] / s.shape[1] for s in samples]  # W/H
        sorted_indices = sorted(range(len(samples)), key=lambda i: aspect_ratios[i])
        
        # Prepare metadata storage
        all_metadata = {
            'original_indices': [],  # Which original sample each output corresponds to
            'resize_scales': [],  # (scale_h, scale_w) for each sample
            'padding': [],  # (left, right, top, bottom) for each sample
            'batch_indices': [],  # (start_idx, end_idx) in flattened output for each batch
        }
        
        batches = []
        current_output_idx = 0
        
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(samples))
            batch_indices = sorted_indices[start_idx:end_idx]
            
            # Get samples for this batch
            batch_samples = [samples[i] for i in batch_indices]
            batch_orig_shapes = [original_shapes[i] for i in batch_indices]
            
            # Find max width in this batch
            max_width = max(s.shape[2] for s in batch_samples)
            target_height = batch_samples[0].shape[1]  # Should all be same height (32)
            
            # Pad each sample to max_width and track metadata
            padded_samples = []
            for sample, orig_shape in zip(batch_samples, batch_orig_shapes):
                C, H, W = sample.shape
                
                # Calculate padding needed (pad right side only)
                pad_right = max_width - W
                pad_left = 0
                pad_top = 0
                pad_bottom = 0
                
                # Apply padding
                if pad_right > 0:
                    # Pad: (left, right, top, bottom)
                    padded = torch.nn.functional.pad(sample, (pad_left, pad_right, pad_top, pad_bottom), value=0)
                else:
                    padded = sample
                
                padded_samples.append(padded)
                
                # Calculate resize scale from original to resized
                scale_h = H / orig_shape[0]
                scale_w = W / orig_shape[1]
                
                # Store metadata
                all_metadata['resize_scales'].append((scale_h, scale_w))
                all_metadata['padding'].append((pad_left, pad_right, pad_top, pad_bottom))
            
            # Store original indices for this batch
            all_metadata['original_indices'].extend(batch_indices)
            
            # Stack into batch
            batch_tensor = torch.stack(padded_samples, dim=0)
            batches.append(batch_tensor)
            
            # Track batch range in output
            batch_start = current_output_idx
            batch_end = current_output_idx + len(batch_samples)
            all_metadata['batch_indices'].extend([batch_idx]*len(batch_samples))
            current_output_idx = batch_end
        
        return batches, all_metadata

    def sample_transforms(self, x: np.ndarray) -> torch.Tensor:
        """
        Transform a single sample. Override this method to add custom preprocessing.
        """
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if x.dtype not in (np.uint8, np.float32, np.float16):
            raise TypeError("unsupported data type for numpy.ndarray")
        tensor = torch.from_numpy(x.copy()).permute(2, 0, 1)
        
        # Resizing - different logic for dynamic vs fixed width
        if self.dynamic_width_batching:
            # Only fix height, preserve aspect ratio for width
            C, H, W = tensor.shape
            target_height = self.output_height
            scale = target_height / H
            target_width = int(W * scale)
            
            # Resize with computed dimensions
            tensor = F.resize(
                tensor, 
                [target_height, target_width],
                interpolation=self.resize.interpolation,
                antialias=self.resize.antialias
            )
        else:
            # Fixed size resize (original behavior)
            tensor = self.resize(tensor)
        
        # Data type
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(dtype=torch.float32).div(255).clip(0, 1)
        else:
            tensor = tensor.to(dtype=torch.float32)

        return tensor

    def __call__(self, x: np.ndarray | list[np.ndarray]) -> list[torch.Tensor]:
        """Prepare document data for model forwarding

        Args:
            x: list of images (np.array) or a single image (np.array) of shape (H, W, C)

        Returns:
            list of page batches (*, C, H, W) ready for model inference
        """
        # Reset metadata
        self.last_metadata = None
        
        # Input type check
        if isinstance(x, np.ndarray):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if x.dtype not in (np.uint8, np.float32, np.float16):
                raise TypeError("unsupported data type for numpy.ndarray")
            tensor = torch.from_numpy(x.copy()).permute(0, 3, 1, 2)

            # Resizing
            if tensor.shape[-2] != self.resize.size[0] or tensor.shape[-1] != self.resize.size[1]:
                tensor = F.resize(
                    tensor, self.resize.size, interpolation=self.resize.interpolation, antialias=self.resize.antialias
                )
            # Data type
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(dtype=torch.float32).div(255).clip(0, 1)
            else:
                tensor = tensor.to(dtype=torch.float32)
            batches = [tensor]

        elif isinstance(x, list) and all(isinstance(sample, np.ndarray) for sample in x):
            # Store original shapes before transformation
            original_shapes = [(sample.shape[0], sample.shape[1]) for sample in x]  # (H, W)
            
            # Sample transform (to tensor, resize)
            samples = list(multithread_exec(self.sample_transforms, x))
            
            # Batching - dynamic or fixed
            if self.dynamic_width_batching:
                batches, metadata = self.batch_inputs_dynamic(samples, original_shapes)
                self.last_metadata = metadata
            else:
                batches = self.batch_inputs(samples)
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        batches = list(multithread_exec(self.normalize, batches))

        return batches
    
    def get_metadata(self) -> dict | None:
        """
        Get metadata from last preprocessing operation.
        
        Returns:
            Metadata dict with indices, scales, padding, or None if not available
        """
        return self.last_metadata
    
    def map_logits_to_original(
        self,
        logits: torch.Tensor,
        metadata: dict | None = None
    ) -> list[dict]:
        """
        Map logits back to original image coordinates.
        
        Args:
            logits: Model output logits of shape (N, seq_len, num_classes)
            metadata: Preprocessing metadata (uses last_metadata if None)
            
        Returns:
            List of dicts, one per original image, containing:
                - 'logits': The logits for this image (seq_len, num_classes)
                - 'original_idx': Original index before sorting
                - 'resize_scale': (scale_h, scale_w) 
                - 'padding': (left, right, top, bottom)
        """
        if metadata is None:
            metadata = self.last_metadata
            
        if metadata is None:
            raise ValueError("No metadata available. Use dynamic_width_batching=True")
        
        # Extract logits for each sample and create mapping
        results = []
        for i in range(len(metadata['original_indices'])):
            results.append({
                'logits': logits[i],  # (seq_len, num_classes)
                'original_idx': metadata['original_indices'][i],
                'resize_scale': metadata['resize_scales'][i],
                'padding': metadata['padding'][i],
            })
        
        # Sort back to original order
        results = sorted(results, key=lambda x: x['original_idx'])
        
        return results


class CustomRecognitionPredictor(nn.Module):
    """
    Custom recognition predictor with access to intermediate outputs.
    
    Extends DocTR's RecognitionPredictor but allows:
    - Access to model logits/out_map
    - Custom preprocessing modifications
    - Modified crop splitting behavior
    - Custom postprocessing
    
    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core recognition architecture
        split_wide_crops: whether to use crop splitting for high aspect ratio crops
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        split_wide_crops: bool = False,
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8  # Critical aspect ratio
        self.overlap_ratio = 0.5  # Ratio of overlap between neighboring crops
        self.target_ar = 6  # Target aspect ratio
        self.batch_size = 128  # Batch size for processing
        # Store last outputs for debugging/analysis
        self.last_raw_output = None
        self.last_logits = None
        self.last_metadata = None

    @torch.inference_mode()
    def forward(
        self,
        crops: Sequence[np.ndarray],
        return_model_output: bool = False,
        **kwargs: Any,
    ) -> list[tuple[str, float]] | dict:
        """
        Forward pass through recognition model.
        
        Args:
            crops: Sequence of cropped word images
            return_model_output: If True, return dict with 'preds' and 'out_map' (logits)
            **kwargs: Additional arguments passed to model
            
        Returns:
            List of (text, confidence) tuples, or dict if return_model_output=True
        """
        if len(crops) == 0:
            if return_model_output:
                return {'preds': [], 'out_map': None}
            return []
        
        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

        # Split crops that are too wide
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(
                crops,  # type: ignore[arg-type]
                self.critical_ar,
                self.target_ar,
                self.overlap_ratio,
            )
            if remapped:
                crops = new_crops

        # Resize & batch them
        processed_batches = self.pre_processor(crops)  # type: ignore[arg-type]

        # Forward it
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(
            self.model, processed_batches, _params.device, _params.dtype
        )
        
        # Get predictions and optionally logits
        if return_model_output:
            raw_outputs = []
            logits_list = []
            
            for batch in processed_batches:
                batch_output = self.model(batch, return_model_output=True, return_preds=True, **kwargs)
                raw_outputs.append(batch_output["preds"])
                
                # Store logits/out_map if available as numpy arrays
                if "out_map" in batch_output:
                    logits_list.append(batch_output["out_map"].cpu().numpy())
            
            # Process predictions
            out = [charseq for batch in raw_outputs for charseq in batch]
            
            # Combine logits
            if logits_list:
                # Concatenate all logits from batches
                all_logits = logits_list
                self.last_logits = all_logits
            else:
                all_logits = None
            
            # Get preprocessing metadata
            metadata = self.pre_processor.get_metadata()
            self.last_metadata = metadata
            
            # Resort logits and predictions back to original order if dynamic batching was used
            if metadata is not None and all_logits is not None:
                # Resort logits to original order
                original_indices = metadata['original_indices']
                batch_indices = metadata['batch_indices']
                
                # Resort predictions to original order
                sorted_preds = [None] * len(original_indices)
                for i, orig_idx in enumerate(original_indices):
                    sorted_preds[orig_idx] = out[i]
                out = sorted_preds
            
            # Remap crops if needed
            if self.split_wide_crops and remapped:
                out = remap_preds(out, crop_map, self.overlap_ratio)
                # Note: logits won't be remapped for split crops - they remain per sub-crop
            
            return {
                'preds': out,
                'out_map': all_logits,
                'metadata': metadata
            }
        else:
            # Standard prediction without logits
            raw = [self.model(batch, return_preds=True, **kwargs)["preds"] for batch in processed_batches]
            self.last_raw_output = raw

            # Process outputs
            out = [charseq for batch in raw for charseq in batch]

            # Remap crops
            if self.split_wide_crops and remapped:
                out = remap_preds(out, crop_map, self.overlap_ratio)

            return out
    
    def get_last_metadata(self) -> dict | None:
        """Get metadata from last forward pass.
        
        Returns:
            Metadata dict with preprocessing info, or None if not available
        """
        return self.last_metadata
    
    def preprocess_only(self, crops: Sequence[np.ndarray]) -> list[torch.Tensor]:
        """
        Only run preprocessing without model inference.
        Useful for debugging or custom processing.
        
        Args:
            crops: Sequence of cropped word images
            
        Returns:
            List of preprocessed batches
        """
        if self.split_wide_crops:
            crops, _, _ = split_crops(
                crops,  # type: ignore[arg-type]
                self.critical_ar,
                self.target_ar,
                self.overlap_ratio,
            )
        
        return self.pre_processor(crops)  # type: ignore[arg-type]
    
    def get_crop_split_info(self, crops: Sequence[np.ndarray]) -> dict:
        """
        Get information about how crops would be split.
        
        Args:
            crops: Sequence of cropped word images
            
        Returns:
            Dict with split information
        """
        if not self.split_wide_crops:
            return {
                'will_split': False,
                'num_original': len(crops),
                'num_after_split': len(crops)
            }
        
        new_crops, crop_map, remapped = split_crops(
            crops,  # type: ignore[arg-type]
            self.critical_ar,
            self.target_ar,
            self.overlap_ratio,
        )
        
        return {
            'will_split': remapped,
            'num_original': len(crops),
            'num_after_split': len(new_crops) if remapped else len(crops),
            'crop_map': crop_map if remapped else None
        }


def create_custom_recognition_predictor(
    arch: Any = 'crnn_vgg16_bn',
    pretrained: bool = False,
    symmetric_pad: bool = False,
    batch_size: int = 128,
    dynamic_width_batching: bool = False,
    **kwargs: Any
) -> CustomRecognitionPredictor:
    """
    Create a custom recognition predictor (matches doctr.models.recognition_predictor).
    
    Args:
        arch: name of the architecture or model itself to use (e.g., 'crnn_vgg16_bn')
        pretrained: If True, returns a model pre-trained on text recognition dataset
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right
        batch_size: number of samples the model processes in parallel
        dynamic_width_batching: if True, batch by aspect ratio with dynamic padding
        **kwargs: optional parameters to be passed to the architecture
        
    Returns:
        CustomRecognitionPredictor instance
    """
    
    # Handle model creation
    if isinstance(arch, str):
        ARCHS = [
            "crnn_vgg16_bn",
            "crnn_mobilenet_v3_small",
            "crnn_mobilenet_v3_large",
            "sar_resnet31",
            "master",
            "vitstr_small",
            "vitstr_base",
            "parseq",
            "viptr_tiny",
        ]
        
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = recognition.__dict__[arch](
            pretrained=pretrained, 
            pretrained_backbone=kwargs.get("pretrained_backbone", True)
        )
    else:
        # Adding the type for torch compiled models to the allowed architectures
        allowed_archs = [
            recognition.CRNN,
            recognition.SAR,
            recognition.MASTER,
            recognition.ViTSTR,
            recognition.PARSeq,
            recognition.VIPTR,
            _CompiledModule,
        ]

        if not isinstance(arch, tuple(allowed_archs)):
            raise ValueError(f"unknown architecture: {type(arch)}")
        _model = arch

    kwargs.pop("pretrained_backbone", None)

    # Get preprocessing parameters from model config
    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", batch_size)
    kwargs["symmetric_pad"] = symmetric_pad
    kwargs["dynamic_width_batching"] = dynamic_width_batching
    
    input_shape = _model.cfg["input_shape"][-2:]
    
    # Create custom preprocessor with model-specific config
    preprocessor = CustomPreProcessor(input_shape, **kwargs)
    
    # Create custom predictor
    predictor = CustomRecognitionPredictor(preprocessor, _model)

    return predictor

