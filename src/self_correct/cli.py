import argparse
import os
import sys

from ..utils import load_rgb_image, resize_image
from .controller import PROJECT_ROOT, SelfCorrectionController
from .evaluator import to_uint8_rgb


def issues_from_feedback(feedback) -> list:
    if not feedback:
        return []
    issues = []
    if "saturation" in feedback:
        issues.append(f"saturation scale x{float(feedback['saturation']):.2f}")
    if "boundary" in feedback:
        issues.append(f"boundary smoothing {float(feedback['boundary']):.2f}")
    semantic = feedback.get("semantic") or {}
    for name, delta in semantic.items():
        issues.append(
            f"semantic[{name}] da={float(delta[0]):.2f}, db={float(delta[1]):.2f}"
        )
    if feedback.get("skin") is not None:
        da, db = feedback["skin"]
        issues.append(f"skin da={float(da):.2f}, db={float(db):.2f}")
    return issues


def print_report(result: dict) -> None:
    print("Self-Correcting Colorization")
    print()
    print(f"Baseline Score: {result['baseline']['overall']:.2f}")
    print()
    for rec in result["iterations"]:
        i = rec["iteration"]
        print(f"Iteration {i}: {rec['overall']:.2f}")
        fb = result["feedback"].get(f"iteration_{i}", {})
        issues = issues_from_feedback(fb)
        if issues:
            print("Issues:")
            for item in issues:
                print(f"  - {item}")
        print()
    print(f"Best Iteration: {result['best_iteration']}")
    print(f"Final Score: {result['best_score']:.2f}")
    print(f"Improvement: {result['absolute_improvement']:.2f}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Iterative Lab-space self-correction around a frozen U-Net. "
            "Does not retrain or update U-Net weights."
        )
    )
    parser.add_argument("--input", required=True, help="Path to a grayscale or RGB image")
    parser.add_argument("--threshold", type=float, default=85.0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--checkpoint", default=None, help="Optional U-Net checkpoint path")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: outputs/self_correction/<image_name>/)",
    )
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Skip DeepLabV3 (no download; semantic/skin metrics disabled)",
    )
    return parser.parse_args(argv)


def output_dir_for(image_path: str, override: str = None) -> str:
    if override:
        return override
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(PROJECT_ROOT, "outputs", "self_correction", stem)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.input):
        print(f"Input image not found: {args.input}", file=sys.stderr)
        return 1

    rgb = to_uint8_rgb(load_rgb_image(args.input))
    rgb = resize_image(rgb, size=(256, 256))
    out_dir = output_dir_for(args.input, args.output_dir)

    controller = SelfCorrectionController(
        device=args.device,
        checkpoint=args.checkpoint,
        load_segmentation=not args.no_segmentation,
    )
    result = controller.self_correct(
        rgb,
        threshold=args.threshold,
        max_iterations=args.max_iterations,
        output_dir=out_dir,
    )
    print_report(result)
    print()
    print(f"Wrote reports to: {out_dir}")
    if not result["checkpoint_loaded"]:
        print(
            "Note: no trained U-Net checkpoint was found; "
            "baseline colorization used randomly initialized weights."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
