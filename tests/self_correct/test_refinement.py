"""Tests for the Lab refinement module."""
import numpy as np
import pytest


def make_lab(h=64, w=64, a_val=10.0, b_val=5.0):
    lab = np.zeros((h, w, 3), dtype=np.float64)
    lab[:, :, 0] = 50.0
    lab[:, :, 1] = a_val
    lab[:, :, 2] = b_val
    return lab


def test_no_change_empty_feedback():
    from src.self_correct.refinement import refine
    lab = make_lab()
    out = refine(lab, {})
    np.testing.assert_array_almost_equal(out, lab)


def test_no_inplace_modification():
    from src.self_correct.refinement import refine
    lab = make_lab(a_val=10.0)
    original = lab.copy()
    refine(lab, {"saturation": 2.0})
    np.testing.assert_array_equal(lab, original)


def test_saturation_boost():
    from src.self_correct.refinement import refine
    lab = make_lab(a_val=10.0, b_val=10.0)
    out = refine(lab, {"saturation": 2.0})
    assert out[0, 0, 1] == pytest.approx(20.0)
    assert out[0, 0, 2] == pytest.approx(20.0)


def test_saturation_reduce():
    from src.self_correct.refinement import refine
    lab = make_lab(a_val=40.0, b_val=40.0)
    out = refine(lab, {"saturation": 0.5})
    assert out[0, 0, 1] == pytest.approx(20.0)


def test_output_clipped_to_lab_range():
    from src.self_correct.refinement import refine
    lab = make_lab(a_val=120.0, b_val=120.0)
    out = refine(lab, {"saturation": 10.0})
    assert out[:, :, 1].max() <= 128.0
    assert out[:, :, 2].max() <= 128.0
    assert out[:, :, 1].min() >= -128.0


def test_l_channel_unchanged():
    from src.self_correct.refinement import refine
    lab = make_lab()
    lab[:, :, 0] = 60.0
    out = refine(lab, {"saturation": 2.0, "boundary": 0.3})
    np.testing.assert_array_almost_equal(out[:, :, 0], lab[:, :, 0])


def test_boundary_smoothing_reduces_ab_variance():
    from src.self_correct.refinement import refine
    rng = np.random.default_rng(42)
    lab = np.zeros((64, 64, 3), dtype=np.float64)
    lab[:, :, 0] = 50.0
    lab[:, :, 1] = rng.uniform(-50, 50, (64, 64))
    lab[:, :, 2] = rng.uniform(-50, 50, (64, 64))
    out = refine(lab, {"boundary": 0.5})
    assert out[:, :, 1].var() < lab[:, :, 1].var()


def test_deterministic():
    from src.self_correct.refinement import refine
    lab = make_lab()
    out1 = refine(lab, {"saturation": 1.5, "boundary": 0.3})
    out2 = refine(lab, {"saturation": 1.5, "boundary": 0.3})
    np.testing.assert_array_equal(out1, out2)


def test_output_dtype_float64():
    from src.self_correct.refinement import refine
    lab = make_lab()
    out = refine(lab, {"saturation": 1.2})
    assert out.dtype == np.float64
