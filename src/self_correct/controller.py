"""Self-correction controller for the colorization pipeline.

Wraps the existing ColorizationUNet with an iterative
evaluate -> feedback -> refine loop. Never retrains the model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..model import ColorizationUNet
from ..utils import (
    denormalize_lab,
    lab_to_rgb,
    preprocess_image,
    resize_image,
    rgb_to_lab,
)
from .evaluator import ImageEvaluator
from .feedback import FeedbackGenerator
from .refinement import refine


class SelfCorrectionController:
    """Iterative self-correcting colorization controller.

    Architecture
    ------------
    Grayscale input
        -> ColorizationUNet (frozen, pre-trained)
        -> Baseline RGB image
        -> ImageEvaluator  (multi-metric scoring)
        -> FeedbackGenerator (per-metric adjustment recommendations)
        -> Lab-space refinement (bilateral edge-aware)
        -> Re-evaluation
        -> Repeat up to max_iterations or until threshold is reached
        -> Return best result (highest score seen)

    Parameters
    ----------
    checkpoint_path:
        Path to the U-Net checkpoint (.pth file).
    device:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    threshold:
        Stop early if overall score exceeds this value (0-100).
    max_iterations:
        Maximum number of refinement iterations (default 3).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        threshold: float = 85.0,
        max_iterations: int = 3,
    ):
        if device == "auto":
            from ..device import get_device
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.max_iterations = max_iterations

        # Load U-Net model (frozen – we never call .train() or backward())
        self.model = ColorizationUNet().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

        # Evaluation and feedback components
        self.evaluator = ImageEvaluator(device=str(self.device))
        self.feedback_gen = FeedbackGenerator(target_overall=threshold)

    # ------------------------------------------------------------------
    def _run_unet(self, gray_rgb: np.ndarray) -> np.ndarray:
        """Run the U-Net on an RGB image (uses only the L* channel).

        Returns a uint8 RGB colorized image at 256x256.
        """
        resized = resize_image(gray_rgb, (256, 256))
        L_norm, _ = preprocess_image(resized)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            ab_pred = self.model(L_tensor)  # [1, 2, 256, 256], Tanh output

        ab_np = ab_pred.squeeze(0).cpu().numpy()  # [2, 256, 256]
        lab = denormalize_lab(L_norm, ab_np)
        rgb_float = lab_to_rgb(lab)
        return (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _get_segmentation(self, rgb_image: np.ndarray) -> Optional[torch.Tensor]:
        """Return the DeepLab segmentation or None if unavailable."""
        if not self.evaluator.seg_available:
            return None
        img_t = (
            torch.from_numpy(rgb_image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            seg_out = self.evaluator.seg_model(img_t)["out"]
        return seg_out.argmax(1)  # [1, H, W]

    # ------------------------------------------------------------------
    def self_correct(self, input_image: np.ndarray) -> Dict[str, Any]:
        """Run the full self-correction pipeline.

        Parameters
        ----------
        input_image:
            Input image as uint8 RGB array. Grayscale inputs (2-D or single-
            channel 3-D) are broadcast to 3 channels before inference.

        Returns
        -------
        Dict with keys:

        - ``baseline_rgb``    – uint8 RGB baseline colorization
        - ``baseline_score``  – float overall score of baseline (0-100)
        - ``baseline_eval``   – full evaluation dict of baseline
        - ``iterations``      – list of per-iteration result dicts
        - ``best_rgb``        – uint8 RGB best result found
        - ``best_score``      – float best overall score seen
        - ``best_iteration``  – int  (0 = baseline, 1..N = iteration number)
        - ``improvement``     – float  best_score - baseline_score  (can be negative)
        - ``pct_improvement`` – float  improvement as percentage of baseline_score
        """
        # Handle grayscale input
        if input_image.ndim == 2:
            gray_rgb = np.stack([input_image] * 3, axis=2)
        elif input_image.ndim == 3 and input_image.shape[2] == 1:
            gray_rgb = np.concatenate([input_image] * 3, axis=2)
        else:
            gray_rgb = input_image

        # Step 1: Baseline colorization via U-Net
        baseline_rgb = self._run_unet(gray_rgb)
        baseline_eval = self.evaluator.evaluate(baseline_rgb)
        baseline_score = baseline_eval["overall"]

        best_rgb = baseline_rgb
        best_score = baseline_score
        best_iteration = 0

        iterations: List[Dict[str, Any]] = []

        # Obtain segmentation once for the baseline image
        seg_pred = self._get_segmentation(baseline_rgb)

        current_rgb = baseline_rgb
        current_eval = baseline_eval

        for i in range(1, self.max_iterations + 1):
            # Early stopping
            if current_eval["overall"] >= self.threshold:
                break

            # Convert current colorized RGB -> Lab for refinement
            current_lab = rgb_to_lab(current_rgb.astype(np.float32) / 255.0)

            # Generate feedback based on current evaluation
            feedback = self.feedback_gen.generate(
                evaluation=current_eval,
                lab_image=current_lab,
                segmentation=seg_pred,
                ref_stats=self.evaluator.ref_stats,
                id_to_name=self.evaluator.id_to_name,
            )

            # Refine in Lab space
            refined_lab = refine(
                lab_image=current_lab,
                feedback=feedback,
                segmentation=seg_pred,
                id_to_name=self.evaluator.id_to_name,
            )

            # Convert back to RGB
            refined_rgb_float = lab_to_rgb(refined_lab)
            refined_rgb = (np.clip(refined_rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)

            # Evaluate refined image
            refined_eval = self.evaluator.evaluate(refined_rgb)
            refined_score = refined_eval["overall"]

            # Update segmentation for next iteration
            seg_pred = self._get_segmentation(refined_rgb)

            # Serialisable feedback summary (tuples -> lists)
            fb_summary: Dict[str, Any] = {}
            for k, v in feedback.items():
                if k == "semantic":
                    fb_summary[k] = {cls: list(delta) for cls, delta in v.items()}
                elif isinstance(v, tuple):
                    fb_summary[k] = list(v)
                else:
                    fb_summary[k] = v

            iter_result: Dict[str, Any] = {
                "iteration": i,
                "rgb": refined_rgb,
                "evaluation": refined_eval,
                "score": refined_score,
                "feedback": fb_summary,
                "issues": _describe_issues(current_eval),
            }
            iterations.append(iter_result)

            # Retain the best result — never force a positive improvement
            if refined_score > best_score:
                best_rgb = refined_rgb
                best_score = refined_score
                best_iteration = i

            current_rgb = refined_rgb
            current_eval = refined_eval

        improvement = best_score - baseline_score
        pct_improvement = (
            (improvement / baseline_score * 100.0) if baseline_score > 0 else 0.0
        )

        return {
            "baseline_rgb": baseline_rgb,
            "baseline_score": baseline_score,
            "baseline_eval": baseline_eval,
            "iterations": iterations,
            "best_rgb": best_rgb,
            "best_score": best_score,
            "best_iteration": best_iteration,
            "improvement": improvement,
            "pct_improvement": pct_improvement,
        }


def _describe_issues(evaluation: Dict[str, Any]) -> List[str]:
    """Convert low metric scores into human-readable issue strings."""
    issues = []
    thresholds = {
        "semantic": (70.0, "Semantic colour consistency is low"),
        "realism": (70.0, "Colour realism (histogram) is low"),
        "boundary": (70.0, "Colour bleeding at boundaries detected"),
        "saturation": (70.0, "Global saturation is outside optimal range"),
    }
    for metric, (thresh, msg) in thresholds.items():
        val = evaluation.get(metric)
        if val is not None and val < thresh:
            issues.append(f"{msg} ({val:.1f}/100)")
    skin = evaluation.get("skin")
    if skin is not None and skin < 70.0:
        issues.append(f"Skin tone plausibility is low ({skin:.1f}/100)")
    return issues
