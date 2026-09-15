"""self_correct package for iterative colorization improvement.

This package does NOT modify the existing U-Net model; it wraps it with an
evaluation-feedback-Lab-refinement loop. U-Net weights are never updated.
"""

from .evaluator import ImageEvaluator, map_evaluable_classes
from .feedback import FeedbackGenerator
from .refinement import refine
from .controller import SelfCorrectionController

__all__ = [
    "ImageEvaluator",
    "FeedbackGenerator",
    "SelfCorrectionController",
    "refine",
    "map_evaluable_classes",
]
