"""Tests for ImageEvaluator."""
import json
import numpy as np
import pytest
from unittest.mock import patch


def make_rgb(h=64, w=64, color=(128, 128, 128)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


@pytest.fixture
def evaluator_no_seg(tmp_path):
    ref = {"person": {"mean": [15.0, 18.0], "cov": [[10.0, 0.0], [0.0, 10.0]]}}
    rf = tmp_path / "stats.json"
    rf.write_text(json.dumps(ref))
    with patch(
        "torchvision.models.segmentation.deeplabv3_resnet101",
        side_effect=RuntimeError("no model in test"),
    ):
        from src.self_correct.evaluator import ImageEvaluator
        ev = ImageEvaluator(device="cpu", ref_stats_path=str(rf))
    return ev


def test_scores_in_range(evaluator_no_seg):
    rgb = make_rgb(64, 64, (100, 150, 200))
    result = evaluator_no_seg.evaluate(rgb)
    for key in ("realism", "boundary", "saturation", "overall"):
        assert 0.0 <= result[key] <= 100.0, f"{key} out of range: {result[key]}"


def test_skin_none_when_no_seg(evaluator_no_seg):
    rgb = make_rgb()
    result = evaluator_no_seg.evaluate(rgb)
    assert result["skin"] is None


def test_semantic_zero_when_no_seg(evaluator_no_seg):
    rgb = make_rgb()
    result = evaluator_no_seg.evaluate(rgb)
    assert result["semantic"] == 0.0


def test_no_nan_inf(evaluator_no_seg):
    rgb = make_rgb(64, 64, (50, 100, 200))
    result = evaluator_no_seg.evaluate(rgb)
    for k, v in result.items():
        if v is not None:
            assert not np.isnan(v), f"NaN in {k}"
            assert not np.isinf(v), f"Inf in {k}"


def test_overall_weighted(evaluator_no_seg):
    rgb = make_rgb()
    result = evaluator_no_seg.evaluate(rgb)
    # Without skin: overall uses semantic + realism + boundary + saturation
    w = evaluator_no_seg.weights
    tw = w["semantic"] + w["realism"] + w["boundary"] + w["saturation"]
    expected = (
        result["semantic"] * w["semantic"]
        + result["realism"] * w["realism"]
        + result["boundary"] * w["boundary"]
        + result["saturation"] * w["saturation"]
    ) / tw
    assert abs(result["overall"] - expected) < 0.01


def test_fully_grey_image(evaluator_no_seg):
    """Fully grey image should not crash."""
    rgb = make_rgb(64, 64, (128, 128, 128))
    result = evaluator_no_seg.evaluate(rgb)
    assert "overall" in result
    assert 0.0 <= result["overall"] <= 100.0


def test_seg_not_available(evaluator_no_seg):
    assert evaluator_no_seg.seg_available is False
    assert evaluator_no_seg.id_to_name == {}
