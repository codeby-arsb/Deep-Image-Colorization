import numpy as np
import pytest

from src.self_correct.refinement import refine


def _base_lab():
    lab = np.zeros((8, 8, 3), dtype=np.float64)
    lab[:, :, 0] = 55.0
    lab[:, :, 1] = 10.0
    lab[:, :, 2] = -12.0
    return lab


def test_preserves_l_and_does_not_mutate_input():
    lab = _base_lab()
    original = lab.copy()
    out = refine(lab, {"saturation": 1.3})
    np.testing.assert_array_equal(lab, original)
    np.testing.assert_allclose(out[:, :, 0], original[:, :, 0])
    np.testing.assert_allclose(out[:, :, 1], original[:, :, 1] * 1.3)
    np.testing.assert_allclose(out[:, :, 2], original[:, :, 2] * 1.3)


def test_applies_region_correction_only_on_mask():
    lab = _base_lab()
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[:4, :] = 15
    out = refine(
        lab,
        {"semantic": {"person": (5.0, -3.0)}},
        segmentation=seg,
        id_to_name={15: "person"},
    )
    np.testing.assert_allclose(out[:4, :, 1], 15.0)
    np.testing.assert_allclose(out[:4, :, 2], -15.0)
    np.testing.assert_allclose(out[4:, :, 1], 10.0)
    np.testing.assert_allclose(out[4:, :, 2], -12.0)
    np.testing.assert_allclose(out[:, :, 0], 55.0)


def test_missing_masks_skip_region_and_skin():
    lab = _base_lab()
    out = refine(
        lab,
        {"semantic": {"person": (9.0, 9.0)}, "skin": (4.0, 4.0)},
        segmentation=None,
        id_to_name={15: "person"},
    )
    np.testing.assert_allclose(out, lab)


def test_skin_only_on_person_region():
    lab = _base_lab()
    seg = np.zeros((8, 8), dtype=np.int64)
    seg[2:6, 2:6] = 15
    out = refine(
        lab,
        {"skin": (2.0, 1.0)},
        segmentation=seg,
        id_to_name={15: "person"},
    )
    np.testing.assert_allclose(out[2:6, 2:6, 1], 12.0)
    np.testing.assert_allclose(out[2:6, 2:6, 2], -11.0)
    np.testing.assert_allclose(out[0, 0, 1], 10.0)


def test_clips_values_and_preserves_shape():
    lab = _base_lab()
    lab[:, :, 1] = 100.0
    out = refine(lab, {"saturation": 2.0})
    assert out.shape == lab.shape
    assert out[:, :, 1].max() <= 128.0
    assert out[:, :, 0].min() >= 0.0
    assert out[:, :, 0].max() <= 100.0


def test_invalid_values_rejected():
    lab = _base_lab()
    lab[0, 0, 1] = np.nan
    with pytest.raises(ValueError):
        refine(lab, {"saturation": 1.1})


def test_boundary_keeps_l_unchanged():
    lab = _base_lab()
    lab[:, 4:, 1] = 40.0
    out = refine(lab, {"boundary": 0.4})
    np.testing.assert_allclose(out[:, :, 0], lab[:, :, 0])
    assert out.shape == lab.shape
