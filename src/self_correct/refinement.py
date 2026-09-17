"""Lab-space image refinement for the self-correcting colorization pipeline.

All operations work in CIE Lab colour space and produce a new array;
the input is never modified in place. Refinement is deterministic.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple


# Valid Lab ranges: L*: [0,100], a*: [-128,128], b*: [-128,128]
_LAB_RANGES: Dict[int, Tuple[float, float]] = {
    0: (0.0, 100.0),
    1: (-128.0, 128.0),
    2: (-128.0, 128.0),
}


def _clip_lab(lab: np.ndarray) -> np.ndarray:
    """Clip each channel to its valid Lab range."""
    result = lab.copy()
    for ch, (lo, hi) in _LAB_RANGES.items():
        result[:, :, ch] = np.clip(result[:, :, ch], lo, hi)
    return result


def _edge_aware_smooth_ab(
    lab: np.ndarray,
    strength: float,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply bilateral filtering to the a*/b* channels.

    Linearly blends between original and smoothed by strength in [0, 1].
    The L* channel is left unchanged. The bilateral filter respects luminance
    edges so colour bleed across high-contrast boundaries is reduced.
    """
    if strength <= 0.0:
        return lab.copy()
    strength = min(1.0, float(strength))

    # Bilateral filter requires uint8 input.
    # Map a* and b* from [-128, 128] to [0, 255].
    a_u8 = np.clip((lab[:, :, 1] + 128.0) / 256.0 * 255.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip((lab[:, :, 2] + 128.0) / 256.0 * 255.0, 0, 255).astype(np.uint8)

    a_smooth = cv2.bilateralFilter(a_u8, d, sigma_color, sigma_space)
    b_smooth = cv2.bilateralFilter(b_u8, d, sigma_color, sigma_space)

    # Map back to float Lab range.
    a_f = a_smooth.astype(np.float64) / 255.0 * 256.0 - 128.0
    b_f = b_smooth.astype(np.float64) / 255.0 * 256.0 - 128.0

    result = lab.copy()
    result[:, :, 1] = (1.0 - strength) * lab[:, :, 1] + strength * a_f
    result[:, :, 2] = (1.0 - strength) * lab[:, :, 2] + strength * b_f
    return result


def refine(
    lab_image: np.ndarray,
    feedback: Dict[str, Any],
    segmentation: Optional[Any] = None,
    id_to_name: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Apply feedback corrections to a Lab image and return a new array.

    Parameters
    ----------
    lab_image:
        H x W x 3 Lab image in float64. Not modified in place.
    feedback:
        Dict produced by FeedbackGenerator.generate. Recognised keys:

        - ``"saturation"``: float scale factor applied to global chroma (a*, b*)
        - ``"semantic"``: ``Dict[str, Tuple[float, float]]`` per-class (da, db) shifts
        - ``"skin"``: ``Tuple[float, float]`` (da, db) for the person region
        - ``"boundary"``: float smoothing strength in [0, 0.5]

    segmentation:
        Optional segmentation tensor (shape [1, H, W], dtype long) from the evaluator.
    id_to_name:
        Mapping ``class_id -> class_name`` from the evaluator.

    Returns
    -------
    np.ndarray
        Refined Lab image clipped to valid Lab ranges. New array, input unchanged.
    """
    result = lab_image.astype(np.float64)

    # 1. Global saturation adjustment (scale a* and b* uniformly)
    if "saturation" in feedback:
        factor = float(feedback["saturation"])
        result[:, :, 1] *= factor
        result[:, :, 2] *= factor

    # 2. Per-class semantic ab shifts
    if "semantic" in feedback and segmentation is not None and id_to_name is not None:
        seg_np = segmentation.squeeze(0).cpu().numpy().astype(np.int64)
        per_class: Dict[str, Tuple[float, float]] = feedback["semantic"]
        for class_id, name in id_to_name.items():
            if name not in per_class:
                continue
            mask = seg_np == class_id
            if not mask.any():
                continue
            da, db = per_class[name]
            result[:, :, 1][mask] += da
            result[:, :, 2][mask] += db

    # 3. Skin-tone correction (person region only)
    if "skin" in feedback and segmentation is not None and id_to_name is not None:
        da, db = feedback["skin"]
        seg_np = segmentation.squeeze(0).cpu().numpy().astype(np.int64)
        person_ids = [cid for cid, name in id_to_name.items() if name == "person"]
        if not person_ids:
            person_ids = [15]  # Confirmed Pascal VOC ID for person.
        person_mask = np.isin(seg_np, person_ids)
        if person_mask.any():
            result[:, :, 1][person_mask] += da
            result[:, :, 2][person_mask] += db

    # 4. Edge-aware boundary smoothing
    if "boundary" in feedback:
        strength = float(feedback["boundary"])
        result = _edge_aware_smooth_ab(result, strength)

    # 5. Clip to valid Lab ranges
    result = _clip_lab(result)
    return result
