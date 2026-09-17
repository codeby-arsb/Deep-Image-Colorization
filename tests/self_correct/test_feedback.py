"""Tests for FeedbackGenerator."""
import numpy as np
import pytest


@pytest.fixture
def generator():
    from src.self_correct.feedback import FeedbackGenerator
    return FeedbackGenerator(target_overall=80.0)


def make_lab(h=64, w=64, a=10.0, b=5.0):
    lab = np.zeros((h, w, 3), dtype=np.float64)
    lab[:, :, 0] = 50.0
    lab[:, :, 1] = a
    lab[:, :, 2] = b
    return lab


GOOD_EVAL = {
    "semantic": 90.0, "realism": 90.0, "boundary": 90.0,
    "saturation": 90.0, "skin": None, "overall": 90.0,
}
LOW_EVAL = {
    "semantic": 10.0, "realism": 10.0, "boundary": 10.0,
    "saturation": 10.0, "skin": None, "overall": 10.0,
}


def test_no_feedback_when_scores_good(generator):
    fb = generator.generate(GOOD_EVAL, make_lab())
    assert fb == {}


def test_saturation_feedback_low_chroma(generator):
    lab = make_lab(a=2.0, b=2.0)  # very low chroma -> should boost
    eval_ = {**LOW_EVAL, "saturation": 10.0}
    fb = generator.generate(eval_, lab)
    assert "saturation" in fb
    assert fb["saturation"] > 1.0, "Expected boost when chroma is very low"


def test_boundary_feedback_generated(generator):
    eval_ = {**GOOD_EVAL, "boundary": 30.0, "overall": 60.0}
    fb = generator.generate(eval_, make_lab())
    assert "boundary" in fb
    assert 0.0 < fb["boundary"] <= 0.5


def test_no_skin_when_evaluation_none(generator):
    eval_ = {**LOW_EVAL, "skin": None}
    fb = generator.generate(eval_, make_lab())
    assert "skin" not in fb


def test_saturation_factor_bounded(generator):
    lab = make_lab(a=0.1, b=0.1)  # near-zero chroma
    eval_ = {**LOW_EVAL}
    fb = generator.generate(eval_, lab)
    if "saturation" in fb:
        assert fb["saturation"] <= generator.max_global_sat_factor


def test_boundary_strength_bounded(generator):
    eval_ = {**LOW_EVAL}
    fb = generator.generate(eval_, make_lab())
    if "boundary" in fb:
        assert fb["boundary"] <= 0.5
