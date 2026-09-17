"""self_correct: iterative self-correcting colorization wrapper.

This package wraps the existing ColorizationUNet with an iterative
evaluate -> feedback -> refine loop. It never modifies or retrains the
U-Net; all corrections are applied in CIE Lab colour space post-inference.
"""
from .evaluator import ImageEvaluator
from .feedback import FeedbackGenerator
from .refinement import refine
from .controller import SelfCorrectionController

__all__ = [
    "ImageEvaluator",
    "FeedbackGenerator",
    "refine",
    "SelfCorrectionController",
]
