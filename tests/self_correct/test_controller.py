"""Tests for SelfCorrectionController."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
import torch


def make_dummy_checkpoint(tmp_path: Path) -> str:
    from src.model import ColorizationUNet
    model = ColorizationUNet()
    ckpt_path = tmp_path / "dummy.pth"
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))
    return str(ckpt_path)


def make_gray_rgb(h=256, w=256):
    return np.full((h, w, 3), 128, dtype=np.uint8)


@pytest.fixture
def controller(tmp_path):
    ckpt = make_dummy_checkpoint(tmp_path)
    with patch(
        "torchvision.models.segmentation.deeplabv3_resnet101",
        side_effect=RuntimeError("skip seg in test"),
    ):
        from src.self_correct.controller import SelfCorrectionController
        ctrl = SelfCorrectionController(
            checkpoint_path=ckpt,
            device="cpu",
            threshold=85.0,
            max_iterations=2,
        )
    return ctrl


def test_self_correct_returns_expected_keys(controller):
    result = controller.self_correct(make_gray_rgb())
    required = (
        "baseline_rgb", "baseline_score", "baseline_eval",
        "iterations", "best_rgb", "best_score",
        "best_iteration", "improvement", "pct_improvement",
    )
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_baseline_rgb_shape(controller):
    result = controller.self_correct(make_gray_rgb())
    assert result["baseline_rgb"].shape == (256, 256, 3)
    assert result["baseline_rgb"].dtype == np.uint8


def test_scores_in_range(controller):
    result = controller.self_correct(make_gray_rgb())
    assert 0.0 <= result["baseline_score"] <= 100.0
    assert 0.0 <= result["best_score"] <= 100.0


def test_best_score_at_least_baseline(controller):
    result = controller.self_correct(make_gray_rgb())
    # Controller never replaces best with a worse result
    assert result["best_score"] >= result["baseline_score"] - 1e-6


def test_max_iterations_respected(controller):
    result = controller.self_correct(make_gray_rgb())
    assert len(result["iterations"]) <= controller.max_iterations


def test_improvement_honest(controller):
    result = controller.self_correct(make_gray_rgb())
    expected = result["best_score"] - result["baseline_score"]
    assert abs(result["improvement"] - expected) < 1e-6


def test_grayscale_2d_input(controller):
    gray2d = np.full((256, 256), 128, dtype=np.uint8)
    result = controller.self_correct(gray2d)
    assert "baseline_rgb" in result


def test_grayscale_3d_single_channel(controller):
    gray3d = np.full((256, 256, 1), 128, dtype=np.uint8)
    result = controller.self_correct(gray3d)
    assert "baseline_rgb" in result


def test_threshold_early_stop(tmp_path):
    """With threshold=0 every image should already pass so 0 iterations run."""
    ckpt = make_dummy_checkpoint(tmp_path)
    with patch(
        "torchvision.models.segmentation.deeplabv3_resnet101",
        side_effect=RuntimeError("skip"),
    ):
        from src.self_correct.controller import SelfCorrectionController
        ctrl = SelfCorrectionController(
            checkpoint_path=ckpt,
            device="cpu",
            threshold=0.0,
            max_iterations=3,
        )
    result = ctrl.self_correct(make_gray_rgb())
    assert len(result["iterations"]) == 0
    assert result["best_iteration"] == 0
