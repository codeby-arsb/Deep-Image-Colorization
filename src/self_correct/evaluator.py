import json
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ..utils import rgb_to_lab

# Pascal VOC 21 class names used by DeepLabV3_ResNet101_Weights.DEFAULT
# (COCO_WITH_VOC_LABELS_V1). Runtime mapping MUST still come from
# weights.meta["categories"] — this list is documentation / test reference only.
PASCAL_VOC_21_CATEGORIES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def map_evaluable_classes(
    categories,
    ref_stat_names,
) -> Dict[int, str]:
    """Map class IDs to names that exist in BOTH weight metadata and ref stats.

    Sky / vegetation / water / road / building are not invented here. If they
    are absent from ``categories``, they are not mapped.
    """
    available = {name: idx for idx, name in enumerate(list(categories))}
    mapping: Dict[int, str] = {}
    for name in ref_stat_names:
        if name in available:
            mapping[available[name]] = name
    return mapping


def lab_histogram(lab_image: np.ndarray, bins: int = 64) -> np.ndarray:
    """Compute a normalized 2-D histogram over the a* and b* channels."""
    a = lab_image[:, :, 1].ravel()
    b = lab_image[:, :, 2].ravel()
    hist, _ = np.histogramdd(
        np.stack([a, b], axis=1),
        bins=bins,
        range=[[-128, 128], [-128, 128]],
    )
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist.ravel()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) with small epsilon to avoid log(0)."""
    eps = 1e-8
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def to_float01_rgb(rgb_image: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb_image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_uint8_rgb(rgb_image: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb_image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


class ImageEvaluator:
    """Evaluates a colorized image and returns a dict of scores (0-100).

    Segmentation model
    ------------------
    torchvision DeepLabV3-ResNet101 with ``DeepLabV3_ResNet101_Weights.DEFAULT``
    (COCO_WITH_VOC_LABELS_V1) exposes the **Pascal VOC 21 classes** only:

        __background__, aeroplane, bicycle, bird, boat, bottle, bus, car,
        cat, chair, cow, diningtable, dog, horse, motorbike, person,
        pottedplant, sheep, sofa, train, tvmonitor.

    Sky, vegetation, water, road, and building are **not** in these weights.
    Semantic evaluation is therefore performed only for classes that appear in
    both ``weights.meta["categories"]`` and ``semantic_ref_stats.json``.
    In the default stats file that is typically just ``person``.
    """

    def __init__(
        self,
        device: str = "auto",
        ref_stats_path: Optional[str] = None,
        reference_hist: Optional[np.ndarray] = None,
        load_segmentation: bool = True,
    ):
        if device == "auto":
            from ..device import get_device

            self.device = get_device()
        else:
            self.device = torch.device(device)

        if ref_stats_path is None:
            ref_stats_path = os.path.join(
                os.path.dirname(__file__), "semantic_ref_stats.json"
            )
        with open(ref_stats_path, "r", encoding="utf-8") as f:
            self.ref_stats = json.load(f)

        if reference_hist is None:
            self.reference_hist = np.ones((64 * 64,), dtype=np.float32) / (64 * 64)
        else:
            self.reference_hist = reference_hist.astype(np.float32)

        self.seg_available = False
        self.seg_model = None
        self.seg_weights = None
        self.seg_preprocess = None
        self.id_to_name: Dict[int, str] = {}
        self.person_class_id: Optional[int] = None
        self.meta_categories = []

        if load_segmentation:
            self._load_segmentation_model()

        self.weights = {
            "semantic": 0.25,
            "realism": 0.20,
            "boundary": 0.20,
            "skin": 0.15,
            "saturation": 0.20,
        }

    def _load_segmentation_model(self) -> None:
        try:
            from torchvision.models.segmentation import (
                DeepLabV3_ResNet101_Weights,
                deeplabv3_resnet101,
            )

            default_weights = DeepLabV3_ResNet101_Weights.DEFAULT
            categories = list(default_weights.meta.get("categories", []))
            self.meta_categories = categories
            self.seg_weights = default_weights
            self.id_to_name = map_evaluable_classes(categories, self.ref_stats.keys())
            for cid, name in self.id_to_name.items():
                if name == "person":
                    self.person_class_id = cid
                    break

            self.seg_model = (
                deeplabv3_resnet101(weights=default_weights).eval().to(self.device)
            )
            try:
                self.seg_preprocess = default_weights.transforms()
            except Exception:
                self.seg_preprocess = None
            self.seg_available = True
        except Exception as exc:
            print(
                f"[ImageEvaluator] Warning: segmentation model unavailable "
                f"({exc}). Semantic and skin metrics will be skipped."
            )
            self.seg_available = False
            self.seg_model = None

    def segment(self, rgb_image: np.ndarray) -> Optional[torch.Tensor]:
        """Return a (H, W) class-id map, or None if segmentation is unavailable."""
        if not self.seg_available or self.seg_model is None:
            return None

        rgb_u8 = to_uint8_rgb(rgb_image)
        height, width = rgb_u8.shape[:2]

        with torch.no_grad():
            if self.seg_preprocess is not None:
                from PIL import Image

                pil = Image.fromarray(rgb_u8)
                inp = self.seg_preprocess(pil)
                if inp.dim() == 3:
                    inp = inp.unsqueeze(0)
                inp = inp.to(self.device)
                logits = self.seg_model(inp)["out"]
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                img_tensor = (
                    torch.from_numpy(to_float01_rgb(rgb_u8))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits = self.seg_model(img_tensor)["out"]
            return logits.argmax(1).squeeze(0).detach().cpu()

    def _semantic_consistency(
        self, lab_image: np.ndarray, segmentation: torch.Tensor
    ) -> float:
        seg_np = np.squeeze(np.asarray(segmentation.cpu() if torch.is_tensor(segmentation) else segmentation)).astype(
            np.int64
        )
        if seg_np.shape[:2] != lab_image.shape[:2]:
            seg_np = cv2.resize(seg_np, (lab_image.shape[1], lab_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        scores = []
        for class_id, name in self.id_to_name.items():
            if name not in self.ref_stats:
                continue
            mask = seg_np == class_id
            if mask.sum() == 0:
                continue
            a_vals = lab_image[:, :, 1][mask]
            b_vals = lab_image[:, :, 2][mask]
            sample_mean = np.stack([a_vals.mean(), b_vals.mean()], axis=0)
            ref = self.ref_stats[name]
            ref_mean = np.array(ref["mean"], dtype=np.float32)
            ref_cov = np.array(ref["cov"], dtype=np.float32)
            diff = sample_mean - ref_mean
            try:
                inv_cov = np.linalg.inv(ref_cov)
                dist = np.sqrt(diff @ inv_cov @ diff)
            except np.linalg.LinAlgError:
                dist = np.linalg.norm(diff)
            score = max(0.0, 100.0 - dist * 10.0)
            scores.append(score)
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _color_realism(self, lab_image: np.ndarray) -> float:
        hist = lab_histogram(lab_image, bins=64)
        kl = kl_divergence(hist, self.reference_hist)
        score = max(0.0, 100.0 - kl * 20.0)
        return float(score)

    def _boundary_consistency(
        self, lab_image: np.ndarray, segmentation: Optional[torch.Tensor] = None
    ) -> float:
        L = lab_image[:, :, 0].astype(np.float32)
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy) + 1e-8
        a = lab_image[:, :, 1]
        b = lab_image[:, :, 2]
        ab = np.stack([a, b], axis=2)
        ab_shift = np.roll(ab, 1, axis=0)
        diff = np.linalg.norm(ab - ab_shift, axis=2)
        ratio = diff / grad_mag
        if segmentation is not None:
            seg_np = np.squeeze(
                np.asarray(segmentation.cpu() if torch.is_tensor(segmentation) else segmentation)
            )
            if seg_np.shape[:2] != L.shape:
                seg_np = cv2.resize(seg_np, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_NEAREST)
            neighbor = np.roll(seg_np, 1, axis=0)
            boundary_mask = seg_np != neighbor
            if boundary_mask.sum() > 0:
                mean_ratio = ratio[boundary_mask].mean()
            else:
                mean_ratio = ratio.mean()
        else:
            mean_ratio = ratio.mean()
        score = max(0.0, 100.0 - mean_ratio * 10.0)
        return float(score)

    def _skin_tone_consistency(
        self, lab_image: np.ndarray, segmentation: torch.Tensor
    ) -> Tuple[float, bool]:
        if self.person_class_id is None:
            return 0.0, False
        seg_np = np.squeeze(
            np.asarray(segmentation.cpu() if torch.is_tensor(segmentation) else segmentation)
        )
        if seg_np.shape[:2] != lab_image.shape[:2]:
            seg_np = cv2.resize(
                seg_np,
                (lab_image.shape[1], lab_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        person_mask = seg_np == self.person_class_id
        if not person_mask.any():
            return 0.0, False
        skin_mean = np.array([15.0, 18.0])
        a_vals = lab_image[:, :, 1][person_mask]
        b_vals = lab_image[:, :, 2][person_mask]
        sample_mean = np.stack([a_vals.mean(), b_vals.mean()], axis=0)
        dist = np.linalg.norm(sample_mean - skin_mean)
        score = max(0.0, 100.0 - dist * 2.0)
        return float(score), True

    def _saturation_score(self, lab_image: np.ndarray) -> float:
        a = lab_image[:, :, 1]
        b = lab_image[:, :, 2]
        chroma = np.sqrt(a ** 2 + b ** 2)
        avg = chroma.mean()
        if avg < 30:
            score = max(0.0, (avg / 30.0) * 100.0)
        elif avg > 70:
            score = max(0.0, ((100 - avg) / 30.0) * 100.0)
        else:
            score = 100.0
        return float(score)

    def _combine_overall(
        self,
        semantic: float,
        realism: float,
        boundary: float,
        saturation: float,
        skin_score: float,
        skin_applicable: bool,
    ) -> float:
        if not skin_applicable:
            total_weight = (
                self.weights["semantic"]
                + self.weights["realism"]
                + self.weights["boundary"]
                + self.weights["saturation"]
            )
            overall = (
                semantic * self.weights["semantic"]
                + realism * self.weights["realism"]
                + boundary * self.weights["boundary"]
                + saturation * self.weights["saturation"]
            ) / total_weight
        else:
            overall = (
                semantic * self.weights["semantic"]
                + realism * self.weights["realism"]
                + boundary * self.weights["boundary"]
                + skin_score * self.weights["skin"]
                + saturation * self.weights["saturation"]
            ) / sum(self.weights.values())
        return float(overall)

    def evaluate(
        self,
        rgb_image: np.ndarray,
        return_aux: bool = False,
    ) -> Any:
        """Run the full evaluation pipeline.

        Returns a dict of metric scores. If ``return_aux`` is True, also
        returns ``{"lab", "segmentation"}``.
        """
        rgb01 = to_float01_rgb(rgb_image)
        lab = rgb_to_lab(rgb01)
        seg_pred = self.segment(rgb_image) if self.seg_available else None

        semantic = (
            self._semantic_consistency(lab, seg_pred) if seg_pred is not None else 0.0
        )
        realism = self._color_realism(lab)
        boundary = self._boundary_consistency(lab, seg_pred)
        if seg_pred is not None:
            skin_score, skin_applicable = self._skin_tone_consistency(lab, seg_pred)
        else:
            skin_score, skin_applicable = 0.0, False
        saturation = self._saturation_score(lab)
        overall = self._combine_overall(
            semantic, realism, boundary, saturation, skin_score, skin_applicable
        )

        def clamp(v):
            value = float(v)
            if not np.isfinite(value):
                return 0.0
            return max(0.0, min(100.0, value))

        result = {
            "semantic": clamp(semantic),
            "realism": clamp(realism),
            "boundary": clamp(boundary),
            "skin": clamp(skin_score) if skin_applicable else None,
            "saturation": clamp(saturation),
            "overall": clamp(overall),
        }
        if return_aux:
            return result, {"lab": lab, "segmentation": seg_pred}
        return result
