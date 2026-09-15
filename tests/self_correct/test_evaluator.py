import numpy as np
import pytest

from src.self_correct.evaluator import (
    PASCAL_VOC_21_CATEGORIES,
    ImageEvaluator,
    map_evaluable_classes,
)


def _make_evaluator():
    return ImageEvaluator(device="cpu", load_segmentation=False)


def test_scores_are_in_0_100():
    ev = _make_evaluator()
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180
    rgb[:, :, 1] = 90
    rgb[:, :, 2] = 40
    result = ev.evaluate(rgb)
    for key in ("semantic", "realism", "boundary", "saturation", "overall"):
        value = result[key]
        assert 0.0 <= value <= 100.0
        assert np.isfinite(value)
    assert result["skin"] is None


def test_weighted_overall_without_skin():
    ev = _make_evaluator()
    overall = ev._combine_overall(40, 80, 60, 100, 0.0, False)
    expected = (40 * 0.25 + 80 * 0.20 + 60 * 0.20 + 100 * 0.20) / 0.85
    assert overall == pytest.approx(expected)


def test_weighted_overall_with_skin():
    ev = _make_evaluator()
    overall = ev._combine_overall(40, 80, 60, 100, 50, True)
    expected = 40 * 0.25 + 80 * 0.20 + 60 * 0.20 + 50 * 0.15 + 100 * 0.20
    assert overall == pytest.approx(expected)


def test_missing_segmentation_sets_skin_none_and_semantic_zero():
    ev = _make_evaluator()
    rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
    result = ev.evaluate(rgb)
    assert ev.seg_available is False
    assert result["skin"] is None
    assert result["semantic"] == 0.0
    assert np.isfinite(result["overall"])


def test_no_nan_inf_in_metrics():
    ev = _make_evaluator()
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    result = ev.evaluate(rgb)
    for value in result.values():
        if value is None:
            continue
        assert np.isfinite(value)


def test_category_mapping_uses_actual_voc_metadata_shape():
    mapping = map_evaluable_classes(
        PASCAL_VOC_21_CATEGORIES,
        ["person", "sky", "vegetation", "water", "road", "building"],
    )
    assert mapping == {15: "person"}
    assert "sky" not in mapping.values()
    assert PASCAL_VOC_21_CATEGORIES[15] == "person"


def test_unknown_metadata_classes_are_not_fabricated():
    mapping = map_evaluable_classes(["__background__", "cat"], ["person", "sky"])
    assert mapping == {}
