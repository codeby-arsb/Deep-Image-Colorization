import numpy as np

from src.self_correct.feedback import FeedbackGenerator


def _good_eval():
    return {
        "semantic": 90.0,
        "realism": 90.0,
        "boundary": 90.0,
        "skin": None,
        "saturation": 95.0,
        "overall": 91.0,
    }


def test_good_scores_produce_no_unnecessary_correction():
    gen = FeedbackGenerator(target_overall=80.0)
    lab = np.zeros((8, 8, 3), dtype=np.float32)
    lab[:, :, 0] = 50
    lab[:, :, 1] = 20
    lab[:, :, 2] = 20
    fb = gen.generate(_good_eval(), lab)
    assert fb == {}


def test_low_saturation_increases_chroma():
    gen = FeedbackGenerator(target_overall=80.0)
    lab = np.zeros((8, 8, 3), dtype=np.float32)
    lab[:, :, 1] = 2.0
    lab[:, :, 2] = 2.0
    fb = gen.generate(
        {"saturation": 10.0, "boundary": 90.0, "semantic": 90.0, "skin": None},
        lab,
    )
    assert "saturation" in fb
    assert fb["saturation"] > 1.0
    assert fb["saturation"] <= gen.max_global_sat_factor


def test_high_chroma_low_score_decreases_chroma():
    gen = FeedbackGenerator(target_overall=80.0)
    lab = np.zeros((8, 8, 3), dtype=np.float32)
    lab[:, :, 1] = 80.0
    lab[:, :, 2] = 80.0
    fb = gen.generate(
        {"saturation": 10.0, "boundary": 90.0, "semantic": 90.0, "skin": None},
        lab,
    )
    assert "saturation" in fb
    assert fb["saturation"] < 1.0
    assert fb["saturation"] >= 1.0 / gen.max_global_sat_factor


def test_low_boundary_generates_boundary_correction():
    gen = FeedbackGenerator(target_overall=80.0)
    lab = np.zeros((4, 4, 3), dtype=np.float32)
    fb = gen.generate(
        {"saturation": 90.0, "boundary": 20.0, "semantic": 90.0, "skin": None},
        lab,
    )
    assert "boundary" in fb
    assert 0.0 < fb["boundary"] <= 0.5


def test_semantic_corrections_only_for_valid_classes():
    gen = FeedbackGenerator(target_overall=80.0)
    lab = np.zeros((6, 6, 3), dtype=np.float32)
    lab[:, :, 1] = 0.0
    lab[:, :, 2] = 0.0
    seg = np.zeros((6, 6), dtype=np.int64)
    seg[:3, :] = 15
    ref_stats = {
        "person": {"mean": [20.0, 15.0], "cov": [[10.0, 0.0], [0.0, 10.0]]},
        "sky": {"mean": [0.0, -10.0], "cov": [[25.0, 0.0], [0.0, 25.0]]},
    }
    fb = gen.generate(
        {"semantic": 10.0, "saturation": 90.0, "boundary": 90.0, "skin": None},
        lab,
        segmentation=seg,
        ref_stats=ref_stats,
        id_to_name={15: "person"},
    )
    assert "semantic" in fb
    assert "person" in fb["semantic"]
    assert "sky" not in fb["semantic"]


def test_skin_correction_absent_when_skin_unavailable():
    gen = FeedbackGenerator()
    lab = np.zeros((4, 4, 3), dtype=np.float32)
    fb = gen.generate(
        {"semantic": 90.0, "saturation": 90.0, "boundary": 90.0, "skin": None},
        lab,
        segmentation=np.zeros((4, 4), dtype=np.int64),
        id_to_name={15: "person"},
    )
    assert "skin" not in fb


def test_correction_values_stay_bounded():
    gen = FeedbackGenerator(max_global_sat_factor=1.5)
    lab = np.zeros((5, 5, 3), dtype=np.float32)
    lab[:, :, 1] = -120
    lab[:, :, 2] = 120
    seg = np.full((5, 5), 15, dtype=np.int64)
    ref_stats = {"person": {"mean": [20.0, 15.0], "cov": [[10.0, 0.0], [0.0, 10.0]]}}
    fb = gen.generate(
        {"semantic": 5.0, "saturation": 5.0, "boundary": 5.0, "skin": 5.0},
        lab,
        segmentation=seg,
        ref_stats=ref_stats,
        id_to_name={15: "person"},
    )
    assert fb["saturation"] <= 1.5
    assert fb["saturation"] >= 1.0 / 1.5
    assert fb["boundary"] <= 0.5
    da, db = fb["semantic"]["person"]
    assert abs(da) <= gen.MAX_DELTA
    assert abs(db) <= gen.MAX_DELTA
    sda, sdb = fb["skin"]
    assert abs(sda) <= gen.MAX_DELTA
    assert abs(sdb) <= gen.MAX_DELTA
