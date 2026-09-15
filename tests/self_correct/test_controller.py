import numpy as np

from src.self_correct.controller import SelfCorrectionController
from src.self_correct.evaluator import ImageEvaluator
from src.self_correct.feedback import FeedbackGenerator


class ScriptedEvaluator(ImageEvaluator):
    def __init__(self, scores):
        super().__init__(device="cpu", load_segmentation=False)
        self._scores = list(scores)
        self.calls = 0

    def evaluate(self, rgb_image, return_aux=False):
        idx = min(self.calls, len(self._scores) - 1)
        self.calls += 1
        result = dict(self._scores[idx])
        lab = np.zeros((8, 8, 3), dtype=np.float64)
        lab[:, :, 0] = 50
        lab[:, :, 1] = 8 + self.calls
        lab[:, :, 2] = 4
        aux = {"lab": lab, "segmentation": None}
        if return_aux:
            return result, aux
        return result


class RecordingFeedback(FeedbackGenerator):
    def __init__(self):
        super().__init__()
        self.generated = []

    def generate(
        self,
        evaluation,
        lab_image,
        segmentation=None,
        ref_stats=None,
        id_to_name=None,
    ):
        fb = {"saturation": 1.05 + 0.01 * len(self.generated)}
        self.generated.append(fb)
        return fb


def _controller(scores, threshold=85.0):
    ev = ScriptedEvaluator(scores)
    fb = RecordingFeedback()

    def colorize_fn(image):
        return np.full((8, 8, 3), 40, dtype=np.uint8)

    ctrl = SelfCorrectionController(
        device="cpu",
        evaluator=ev,
        feedback_generator=fb,
        colorize_fn=colorize_fn,
        load_model=False,
        load_segmentation=False,
    )
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    result = ctrl.self_correct(rgb, threshold=threshold, max_iterations=3)
    return result, ev, fb


def _score(overall, **extra):
    base = {
        "semantic": 0.0,
        "realism": 50.0,
        "boundary": 50.0,
        "skin": None,
        "saturation": 50.0,
        "overall": overall,
    }
    base.update(extra)
    return base


def test_threshold_stopping():
    result, ev, fb = _controller(
        [_score(70), _score(90), _score(91)],
        threshold=85,
    )
    assert result["iterations_run"] == 1
    assert len(result["iterations"]) == 1
    assert ev.calls == 2
    assert len(fb.generated) == 1


def test_max_iteration_stopping():
    result, ev, fb = _controller(
        [_score(70), _score(71), _score(72), _score(73)],
        threshold=85,
    )
    assert result["iterations_run"] == 3
    assert len(result["iterations"]) == 3
    assert ev.calls == 4


def test_best_score_selection_last_is_best():
    result, _, _ = _controller(
        [_score(72), _score(76), _score(74), _score(81)],
        threshold=100,
    )
    assert result["best_iteration"] == 3
    assert result["best_score"] == 81


def test_worse_iteration_does_not_replace_best():
    result, _, _ = _controller(
        [_score(72), _score(80), _score(77), _score(79)],
        threshold=100,
    )
    assert result["best_iteration"] == 1
    assert result["best_score"] == 80


def test_iteration_logging_and_baseline_comparison():
    result, _, fb = _controller(
        [_score(72), _score(76), _score(74), _score(81)],
        threshold=100,
    )
    assert result["baseline_score"] == 72
    assert result["absolute_improvement"] == 9
    assert result["percentage_improvement"] == (9 / 72) * 100
    assert [r["iteration"] for r in result["iterations"]] == [1, 2, 3]
    assert "iteration_1" in result["feedback"]
    assert len(fb.generated) == 3
    assert result["unet_weights_updated"] is False


def test_negative_improvement_is_allowed():
    result, _, _ = _controller(
        [_score(80), _score(70), _score(71), _score(69)],
        threshold=100,
    )
    assert result["best_iteration"] == 2
    assert result["best_score"] == 71
    assert result["absolute_improvement"] == -9
