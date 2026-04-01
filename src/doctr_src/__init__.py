"""DocTR-specific components for character extraction.

This package wraps DocTR's detection and recognition predictors with
custom extensions (logit access, dynamic-width batching) and provides
utilities for converting CRNN logits into character positions.

Modules
-------
predictor
    :class:`CustomOCRPredictor` — OCR pipeline combining detection +
    recognition with access to intermediate logits.
recognition
    :class:`CustomRecognitionPredictor` — recognition model wrapper with
    dynamic-width batching and logit output.
    :func:`create_custom_recognition_predictor` — factory function.
utils
    :func:`rcnn_positions` — CTC-decode CRNN logits into pixel-level
    character boundary positions.
    :func:`get_word_logits` — extract per-word logits from batch output.
    :func:`char_segment_word` — end-to-end word-level segmentation using
    logit-initialised boxes.
    :func:`gamma_correction` — adaptive gamma correction for page images.
"""
