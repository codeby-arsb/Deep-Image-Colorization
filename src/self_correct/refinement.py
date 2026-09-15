from typing import Any, Dict, Optional

import cv2
import numpy as np

from .feedback import as_seg_np

L_MIN, L_MAX = 0.0, 100.0
AB_MIN, AB_MAX = -128.0, 128.0


def _clip_lab(lab: np.ndarray) -> np.ndarray:
    out = lab.copy()
    out[:, :, 0] = np.clip(out[:, :, 0], L_MIN, L_MAX)
    out[:, :, 1] = np.clip(out[:, :, 1], AB_MIN, AB_MAX)
    out[:, :, 2] = np.clip(out[:, :, 2], AB_MIN, AB_MAX)
    return out


def _name_to_ids(id_to_name: Optional[Dict[int, str]]) -> Dict[str, int]:
    if not id_to_name:
        return {}
    return {name: cid for cid, name in id_to_name.items()}


def refine(
    lab_image: np.ndarray,
    feedback: Optional[Dict[str, Any]] = None,
    segmentation=None,
    id_to_name: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """Deterministic CIE Lab refinement. Never mutates ``lab_image``.

    Uses the same Lab convention as ``src.utils.rgb_to_lab`` / ``lab_to_rgb``
    (skimage CIE Lab: L* in [0, 100], a*/b* typically in [-128, 128]).
    L* is not modified except for safety clipping of invalid values.
    """
    if lab_image is None:
        raise ValueError("lab_image is required")
    lab = np.array(lab_image, dtype=np.float64, copy=True)
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError(f"lab_image must have shape (H, W, 3), got {lab.shape}")
    if not np.isfinite(lab).all():
        raise ValueError("lab_image contains NaN or inf")

    original_l = lab[:, :, 0].copy()
    feedback = feedback or {}
    seg_np = as_seg_np(segmentation)
    if seg_np is not None and seg_np.shape[:2] != lab.shape[:2]:
        seg_np = cv2.resize(
            seg_np,
            (lab.shape[1], lab.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    name_to_id = _name_to_ids(id_to_name)

    sat = feedback.get("saturation")
    if sat is not None:
        factor = float(sat)
        if not np.isfinite(factor):
            raise ValueError("saturation factor is not finite")
        lab[:, :, 1] *= factor
        lab[:, :, 2] *= factor

    semantic = feedback.get("semantic") or {}
    if semantic and seg_np is not None:
        for name, delta in semantic.items():
            cid = name_to_id.get(name)
            if cid is None:
                continue
            mask = seg_np == cid
            if not mask.any():
                continue
            da, db = float(delta[0]), float(delta[1])
            if not (np.isfinite(da) and np.isfinite(db)):
                continue
            lab[:, :, 1][mask] += da
            lab[:, :, 2][mask] += db

    skin = feedback.get("skin")
    if skin is not None and seg_np is not None:
        cid = name_to_id.get("person")
        if cid is not None:
            mask = seg_np == cid
            if mask.any():
                da, db = float(skin[0]), float(skin[1])
                if np.isfinite(da) and np.isfinite(db):
                    lab[:, :, 1][mask] += da
                    lab[:, :, 2][mask] += db

    boundary = feedback.get("boundary")
    if boundary is not None:
        strength = float(np.clip(boundary, 0.0, 1.0))
        if strength > 0:
            a = lab[:, :, 1].astype(np.float32)
            b = lab[:, :, 2].astype(np.float32)
            filtered_a = cv2.bilateralFilter(a, d=5, sigmaColor=25.0, sigmaSpace=5)
            filtered_b = cv2.bilateralFilter(b, d=5, sigmaColor=25.0, sigmaSpace=5)
            lab[:, :, 1] = (1.0 - strength) * a + strength * filtered_a
            lab[:, :, 2] = (1.0 - strength) * b + strength * filtered_b

    lab[:, :, 0] = original_l
    lab = _clip_lab(lab)
    if lab.shape != tuple(lab_image.shape):
        raise RuntimeError("refinement changed image dimensions")
    return lab
