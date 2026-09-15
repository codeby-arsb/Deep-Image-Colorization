from typing import Any, Dict, Optional, Tuple

import numpy as np


def as_seg_np(segmentation) -> Optional[np.ndarray]:
    """Convert a torch tensor or ndarray segmentation map to squeezed int64."""
    if segmentation is None:
        return None
    if hasattr(segmentation, "detach"):
        arr = segmentation.detach().cpu().numpy()
    else:
        arr = np.asarray(segmentation)
    return np.squeeze(arr).astype(np.int64)


class FeedbackGenerator:
    """Generate corrective feedback based on evaluation scores.

    This module does not update U-Net weights. It only describes Lab-space
    adjustments for the refinement stage.
    """

    MAX_DELTA = 40.0

    def __init__(
        self,
        target_overall: float = 80.0,
        metric_thresholds: Optional[Dict[str, float]] = None,
        max_global_sat_factor: float = 1.5,
    ):
        self.target_overall = target_overall
        self.metric_thresholds = metric_thresholds or {}
        self.max_global_sat_factor = max_global_sat_factor

    def _default_thresh(self, metric: str) -> float:
        return self.metric_thresholds.get(metric, self.target_overall - 10.0)

    def _clip_delta(self, delta_a: float, delta_b: float) -> Tuple[float, float]:
        return (
            float(np.clip(delta_a, -self.MAX_DELTA, self.MAX_DELTA)),
            float(np.clip(delta_b, -self.MAX_DELTA, self.MAX_DELTA)),
        )

    def generate(
        self,
        evaluation: Dict[str, Any],
        lab_image: np.ndarray,
        segmentation=None,
        ref_stats: Optional[Dict[str, Any]] = None,
        id_to_name: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """Return a dict describing per-metric Lab adjustments."""
        feedback: Dict[str, Any] = {}
        id_to_name = id_to_name or {}
        seg_np = as_seg_np(segmentation)

        if seg_np is not None and ref_stats is not None and id_to_name:
            sem_score = evaluation.get("semantic", 0.0)
            if sem_score is not None and sem_score < self._default_thresh("semantic"):
                per_class: Dict[str, Tuple[float, float]] = {}
                for cid, name in id_to_name.items():
                    if name not in ref_stats:
                        continue
                    mask = seg_np == cid
                    if not mask.any():
                        continue
                    a_vals = lab_image[:, :, 1][mask]
                    b_vals = lab_image[:, :, 2][mask]
                    sample_mean = np.stack([a_vals.mean(), b_vals.mean()], axis=0)
                    ref_mean = np.array(ref_stats[name]["mean"], dtype=np.float32)
                    delta = ref_mean - sample_mean
                    per_class[name] = self._clip_delta(float(delta[0]), float(delta[1]))
                if per_class:
                    feedback["semantic"] = per_class

        sat_score = evaluation.get("saturation", 0.0)
        if sat_score is not None and sat_score < self._default_thresh("saturation"):
            a = lab_image[:, :, 1]
            b = lab_image[:, :, 2]
            chroma = np.sqrt(a ** 2 + b ** 2)
            avg = float(chroma.mean())
            target = 50.0
            factor = target / (avg + 1e-8)
            lo = 1.0 / self.max_global_sat_factor
            factor = float(np.clip(factor, lo, self.max_global_sat_factor))
            feedback["saturation"] = factor

        if evaluation.get("skin") is not None:
            skin_score = evaluation["skin"]
            if (
                skin_score < self._default_thresh("skin")
                and seg_np is not None
                and id_to_name
            ):
                person_ids = [cid for cid, name in id_to_name.items() if name == "person"]
                if person_ids:
                    person_mask = np.isin(seg_np, person_ids)
                    if person_mask.any():
                        a_vals = lab_image[:, :, 1][person_mask]
                        b_vals = lab_image[:, :, 2][person_mask]
                        sample_mean = np.stack([a_vals.mean(), b_vals.mean()], axis=0)
                        skin_ref = np.array([15.0, 18.0], dtype=np.float32)
                        delta = skin_ref - sample_mean
                        feedback["skin"] = self._clip_delta(
                            float(delta[0]), float(delta[1])
                        )

        boundary_score = evaluation.get("boundary", 0.0)
        if boundary_score is not None and boundary_score < self._default_thresh("boundary"):
            deficit = self._default_thresh("boundary") - boundary_score
            smoothing = min(0.5, max(0.0, deficit / 100.0))
            feedback["boundary"] = float(smoothing)

        return feedback
