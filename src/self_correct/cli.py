"""Command-line interface for the self-correcting colorization pipeline.

Usage::

    python -m src.self_correct.cli \
        --input <image> \
        [--threshold 85] \
        [--max-iterations 3] \
        [--device auto] \
        [--checkpoint path/to/best.pth]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _find_checkpoint() -> str:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "../../outputs/checkpoints/best.pth",
        here / "../../outputs/checkpoints/latest.pth",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(
        "No checkpoint found at outputs/checkpoints/best.pth or latest.pth. "
        "Train the model first or pass --checkpoint."
    )


def _save_image(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(str(path))


def _plot_score_progression(scores: List[float], labels: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(scores)), scores, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Overall Score (0-100)", fontsize=11)
    ax.set_title("Self-Correction Score Progression", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    for i, s in enumerate(scores):
        ax.annotate(
            f"{s:.1f}",
            (i, s),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def _plot_comparison(baseline: np.ndarray, final: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(baseline)
    axes[0].set_title("Baseline (U-Net only)", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(final)
    axes[1].set_title("Self-Corrected (Best Result)", fontsize=12)
    axes[1].axis("off")
    fig.suptitle("Self-Correcting Colorization", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    from .controller import SelfCorrectionController

    checkpoint = getattr(args, "checkpoint", None) or _find_checkpoint()

    img_path = Path(args.input)
    if not img_path.exists():
        print(f"ERROR: Input image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    input_img = np.array(Image.open(str(img_path)).convert("RGB"))

    img_name = img_path.stem
    out_dir = Path("outputs") / "self_correction" / img_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Self-Correcting Colorization")
    print("=" * 40)
    print(f"Input       : {img_path}")
    print(f"Output dir  : {out_dir}")
    print(f"Device      : {args.device}")
    print(f"Threshold   : {args.threshold}  |  Max iterations: {args.max_iterations}")
    print()

    controller = SelfCorrectionController(
        checkpoint_path=checkpoint,
        device=args.device,
        threshold=args.threshold,
        max_iterations=args.max_iterations,
    )

    result = controller.self_correct(input_img)

    # Save baseline
    _save_image(result["baseline_rgb"], out_dir / "baseline.png")

    print(f"Baseline Score: {result['baseline_score']:.2f}")
    print()

    scores = [result["baseline_score"]]
    labels = ["Baseline"]

    all_evals: dict = {"baseline": result["baseline_eval"]}
    all_feedbacks: dict = {}

    for it in result["iterations"]:
        n = it["iteration"]
        _save_image(it["rgb"], out_dir / f"iteration_{n}.png")
        all_evals[f"iteration_{n}"] = it["evaluation"]
        all_feedbacks[f"iteration_{n}"] = it["feedback"]
        scores.append(it["score"])
        labels.append(f"Iter {n}")

        print(f"Iteration {n}: {it['score']:.2f}")
        if it["issues"]:
            print("  Issues:")
            for issue in it["issues"]:
                print(f"    - {issue}")
        print()

    # Save final (best)
    _save_image(result["best_rgb"], out_dir / "final.png")

    # Serialise evaluation results
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(all_evals, f, indent=2)

    with open(out_dir / "feedback.json", "w") as f:
        json.dump(all_feedbacks, f, indent=2)

    _plot_score_progression(scores, labels, out_dir / "score_progression.png")
    _plot_comparison(result["baseline_rgb"], result["best_rgb"], out_dir / "comparison.png")

    print(f"Best Iteration : {result['best_iteration']}")
    print(f"Final Score    : {result['best_score']:.2f}")
    sign = "+" if result["improvement"] >= 0 else ""
    print(f"Improvement    : {sign}{result['improvement']:.2f} "
          f"({sign}{result['pct_improvement']:.1f}%)")
    print()
    print("Output files:")
    for fp in sorted(out_dir.iterdir()):
        if fp.name != ".gitkeep":
            print(f"  {fp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-correcting colorization pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument(
        "--threshold", type=float, default=85.0,
        help="Score threshold for early stopping (0-100)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Maximum number of refinement iterations"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Compute device: auto | cpu | cuda"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to U-Net checkpoint (auto-detected if omitted)"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
