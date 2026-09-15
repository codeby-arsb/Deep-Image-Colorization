import json

import cv2
import numpy as np

from src.self_correct import cli
from src.self_correct.controller import SelfCorrectionController


def test_cli_smoke(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :] = [80, 80, 80]
    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    out_dir = tmp_path / "out"

    class FakeController(SelfCorrectionController):
        def __init__(self, *args, **kwargs):
            dummy = np.full((32, 32, 3), 90, dtype=np.uint8)
            super().__init__(
                device="cpu",
                colorize_fn=lambda img: dummy,
                load_model=False,
                load_segmentation=False,
            )

        def self_correct(
            self,
            rgb_image,
            threshold=85.0,
            max_iterations=3,
            output_dir=None,
            image_name=None,
        ):
            result = {
                "baseline": {
                    "semantic": 0.0,
                    "realism": 40.0,
                    "boundary": 50.0,
                    "skin": None,
                    "saturation": 45.0,
                    "overall": 44.0,
                },
                "iterations": [
                    {
                        "iteration": 1,
                        "semantic": 0.0,
                        "realism": 42.0,
                        "boundary": 52.0,
                        "skin": None,
                        "saturation": 48.0,
                        "overall": 46.0,
                    }
                ],
                "best_iteration": 1,
                "best_score": 46.0,
                "absolute_improvement": 2.0,
                "percentage_improvement": (2.0 / 44.0) * 100,
                "feedback": {"iteration_1": {"saturation": 1.1}},
                "baseline_rgb": np.full((32, 32, 3), 70, dtype=np.uint8),
                "best_rgb": np.full((32, 32, 3), 90, dtype=np.uint8),
                "gray_rgb": np.full((32, 32, 3), 80, dtype=np.uint8),
                "iteration_images": [np.full((32, 32, 3), 90, dtype=np.uint8)],
                "checkpoint_loaded": False,
                "checkpoint_path": None,
                "unet_weights_updated": False,
                "iterations_run": 1,
            }
            if output_dir:
                SelfCorrectionController._write_outputs(self, output_dir, result)
            return result

    monkeypatch.setattr(cli, "SelfCorrectionController", FakeController)
    code = cli.main(
        [
            "--input",
            str(image_path),
            "--threshold",
            "85",
            "--max-iterations",
            "3",
            "--device",
            "cpu",
            "--output-dir",
            str(out_dir),
            "--no-segmentation",
        ]
    )
    assert code == 0
    assert (out_dir / "baseline.png").is_file()
    assert (out_dir / "iteration_1.png").is_file()
    assert not (out_dir / "iteration_2.png").is_file()
    assert (out_dir / "final.png").is_file()
    assert (out_dir / "evaluation.json").is_file()
    assert (out_dir / "feedback.json").is_file()
    assert (out_dir / "score_progression.png").is_file()
    assert (out_dir / "comparison.png").is_file()
    payload = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["best_iteration"] == 1
    assert payload["baseline"]["overall"] == 44.0
