import json
import os
from typing import Any, Callable, Dict, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..device import get_device
from ..model import ColorizationUNet
from ..utils import (
    lab_to_rgb,
    normalize_lab,
    reconstruct_rgb,
    resize_image,
    rgb_to_lab,
)
from .evaluator import ImageEvaluator, to_float01_rgb, to_uint8_rgb
from .feedback import FeedbackGenerator
from .refinement import refine


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if hasattr(obj, "item"):
        try:
            return json_ready(obj.item())
        except Exception:
            return str(obj)
    return obj


def default_checkpoint_candidates() -> List[str]:
    return [
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best.pth"),
        os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "latest.pth"),
        os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1", "checkpoints", "best.pth"),
        os.path.join(PROJECT_ROOT, "models", "best.pth"),
        os.path.join(PROJECT_ROOT, "models", "colorization_unet.pth"),
    ]


def save_rgb(path: str, rgb: np.ndarray) -> None:
    rgb_u8 = to_uint8_rgb(rgb)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


def grayscale_rgb_from_image(rgb: np.ndarray) -> np.ndarray:
    rgb01 = to_float01_rgb(rgb)
    lab = rgb_to_lab(rgb01)
    L = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)
    gray = np.stack([L, L, L], axis=-1)
    return to_uint8_rgb(gray)


class SelfCorrectionController:
    """Iterative Lab-space self-correction around a frozen U-Net.

    The U-Net is run **once**. Subsequent iterations refine the Lab prediction.
    U-Net weights are never updated.
    """

    def __init__(
        self,
        device: str = "auto",
        checkpoint: Optional[str] = None,
        evaluator: Optional[ImageEvaluator] = None,
        feedback_generator: Optional[FeedbackGenerator] = None,
        colorize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        load_model: bool = True,
        load_segmentation: bool = True,
    ):
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self.colorize_fn = colorize_fn
        self.model = None
        self.checkpoint_path = None
        self.checkpoint_loaded = False

        if colorize_fn is None and load_model:
            self.model = ColorizationUNet().to(self.device)
            self.model.eval()
            path = checkpoint
            if path is None:
                for candidate in default_checkpoint_candidates():
                    if os.path.isfile(candidate):
                        path = candidate
                        break
            if path and os.path.isfile(path):
                self._load_checkpoint(path)

        self.evaluator = evaluator or ImageEvaluator(
            device=str(self.device),
            load_segmentation=load_segmentation,
        )
        self.feedback_generator = feedback_generator or FeedbackGenerator()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        chk = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(chk, dict) and "model_state_dict" in chk:
            self.model.load_state_dict(chk["model_state_dict"])
        elif isinstance(chk, dict):
            self.model.load_state_dict(chk)
        else:
            raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_loaded = True

    def colorize(self, rgb_image: np.ndarray) -> np.ndarray:
        """Run the frozen U-Net once. Returns RGB uint8 (256x256)."""
        if self.colorize_fn is not None:
            return to_uint8_rgb(self.colorize_fn(rgb_image))

        if self.model is None:
            raise RuntimeError("No colorization model is available")

        resized = resize_image(to_uint8_rgb(rgb_image), size=(256, 256))
        rgb01 = to_float01_rgb(resized)
        lab = rgb_to_lab(rgb01)
        L_norm, _ = normalize_lab(lab)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            pred_ab = self.model(L_tensor)
        pred_ab_np = pred_ab.squeeze(0).cpu().numpy().astype(np.float32)
        pred_rgb = reconstruct_rgb(L_norm, pred_ab_np)
        return to_uint8_rgb(pred_rgb)

    def self_correct(
        self,
        rgb_image: np.ndarray,
        threshold: float = 85.0,
        max_iterations: int = 3,
        output_dir: Optional[str] = None,
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_rgb = to_uint8_rgb(rgb_image)
        gray_rgb = grayscale_rgb_from_image(source_rgb)
        if gray_rgb.shape[0] != 256 or gray_rgb.shape[1] != 256:
            gray_rgb = resize_image(gray_rgb, size=(256, 256))

        baseline_rgb = self.colorize(gray_rgb)
        baseline_eval, aux = self.evaluator.evaluate(baseline_rgb, return_aux=True)
        current_lab = np.array(aux["lab"], dtype=np.float64, copy=True)
        original_l = current_lab[:, :, 0].copy()
        current_seg = aux["segmentation"]
        current_rgb = baseline_rgb
        current_eval = baseline_eval

        best_image = baseline_rgb.copy()
        best_score = float(baseline_eval["overall"])
        best_iteration = 0
        best_eval = dict(baseline_eval)

        iteration_records: List[Dict[str, Any]] = []
        feedback_log: Dict[str, Any] = {}
        iteration_images: List[np.ndarray] = []

        ran_until = 0
        if best_score < threshold:
            for i in range(1, max_iterations + 1):
                feedback = self.feedback_generator.generate(
                    current_eval,
                    current_lab,
                    segmentation=current_seg,
                    ref_stats=self.evaluator.ref_stats,
                    id_to_name=self.evaluator.id_to_name,
                )
                feedback_log[f"iteration_{i}"] = json_ready(feedback)

                current_lab = refine(
                    current_lab,
                    feedback,
                    segmentation=current_seg,
                    id_to_name=self.evaluator.id_to_name,
                )
                current_lab[:, :, 0] = original_l
                current_rgb = to_uint8_rgb(lab_to_rgb(current_lab))
                current_eval, aux = self.evaluator.evaluate(current_rgb, return_aux=True)
                current_lab = np.array(aux["lab"], dtype=np.float64, copy=True)
                current_lab[:, :, 0] = original_l
                current_seg = aux["segmentation"]

                record = {"iteration": i, **current_eval}
                iteration_records.append(record)
                iteration_images.append(current_rgb.copy())
                ran_until = i

                score = float(current_eval["overall"])
                if i == 1 or score > best_score:
                    best_score = score
                    best_image = current_rgb.copy()
                    best_iteration = i
                    best_eval = dict(current_eval)

                if score >= threshold:
                    break

        baseline_score = float(baseline_eval["overall"])
        abs_improvement = best_score - baseline_score
        pct_improvement = (
            (abs_improvement / baseline_score) * 100.0 if baseline_score != 0 else 0.0
        )

        result = {
            "baseline": baseline_eval,
            "iterations": iteration_records,
            "best_iteration": best_iteration,
            "best_score": best_score,
            "best_evaluation": best_eval,
            "baseline_score": baseline_score,
            "absolute_improvement": abs_improvement,
            "percentage_improvement": pct_improvement,
            "baseline_rgb": baseline_rgb,
            "best_rgb": best_image,
            "gray_rgb": gray_rgb,
            "iteration_images": iteration_images,
            "feedback": feedback_log,
            "threshold": threshold,
            "max_iterations": max_iterations,
            "iterations_run": ran_until,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_loaded": self.checkpoint_loaded,
            "unet_weights_updated": False,
        }

        if output_dir:
            self._write_outputs(output_dir, result, image_name=image_name)
        return result

    def _write_outputs(
        self,
        output_dir: str,
        result: Dict[str, Any],
        image_name: Optional[str] = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        save_rgb(os.path.join(output_dir, "baseline.png"), result["baseline_rgb"])
        for idx, img in enumerate(result["iteration_images"], start=1):
            save_rgb(os.path.join(output_dir, f"iteration_{idx}.png"), img)
        save_rgb(os.path.join(output_dir, "final.png"), result["best_rgb"])

        evaluation_payload = {
            "baseline": json_ready(result["baseline"]),
            "iterations": json_ready(result["iterations"]),
            "best_iteration": result["best_iteration"],
            "best_score": json_ready(result["best_score"]),
            "absolute_improvement": json_ready(result["absolute_improvement"]),
            "percentage_improvement": json_ready(result["percentage_improvement"]),
            "unet_weights_updated": False,
            "checkpoint_loaded": result["checkpoint_loaded"],
            "checkpoint_path": result["checkpoint_path"],
        }
        with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump(evaluation_payload, f, indent=2)

        with open(os.path.join(output_dir, "feedback.json"), "w", encoding="utf-8") as f:
            json.dump(json_ready(result["feedback"]), f, indent=2)

        self._plot_scores(output_dir, result)
        self._plot_comparison(output_dir, result)

    def _plot_scores(self, output_dir: str, result: Dict[str, Any]) -> None:
        xs = [0]
        ys = [float(result["baseline"]["overall"])]
        for rec in result["iterations"]:
            xs.append(int(rec["iteration"]))
            ys.append(float(rec["overall"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o", color="#2563eb")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Overall Score")
        ax.set_title("Self-Correction Score Progression")
        ax.set_xticks(xs)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "score_progression.png"), dpi=120)
        plt.close(fig)

    def _plot_comparison(self, output_dir: str, result: Dict[str, Any]) -> None:
        panels = [
            (result["gray_rgb"], "Original B&W"),
            (result["baseline_rgb"], "Baseline U-Net"),
            (result["best_rgb"], "Final Self-Corrected"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(to_uint8_rgb(img))
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle("Self-Correcting Colorization")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=120)
        plt.close(fig)
