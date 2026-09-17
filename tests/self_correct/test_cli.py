"""Tests for the CLI module."""
import json
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


@pytest.fixture
def cli_env(tmp_path):
    from PIL import Image
    img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    img_path = tmp_path / "test_input.png"
    img.save(str(img_path))
    ckpt = make_dummy_checkpoint(tmp_path)
    return {"img_path": str(img_path), "ckpt": ckpt, "tmp_path": tmp_path}


def _run_cli(cli_env, tmp_path, monkeypatch, max_iterations=1):
    monkeypatch.chdir(tmp_path)
    with patch(
        "torchvision.models.segmentation.deeplabv3_resnet101",
        side_effect=RuntimeError("skip seg in test"),
    ):
        from src.self_correct import cli
        import argparse
        args = argparse.Namespace(
            input=cli_env["img_path"],
            threshold=85.0,
            max_iterations=max_iterations,
            device="cpu",
            checkpoint=cli_env["ckpt"],
        )
        cli.run(args)
    return tmp_path / "outputs" / "self_correction" / "test_input"


def test_cli_produces_baseline(cli_env, tmp_path, monkeypatch):
    out_dir = _run_cli(cli_env, tmp_path, monkeypatch)
    assert (out_dir / "baseline.png").exists()


def test_cli_produces_final(cli_env, tmp_path, monkeypatch):
    out_dir = _run_cli(cli_env, tmp_path, monkeypatch)
    assert (out_dir / "final.png").exists()


def test_cli_produces_evaluation_json(cli_env, tmp_path, monkeypatch):
    out_dir = _run_cli(cli_env, tmp_path, monkeypatch)
    assert (out_dir / "evaluation.json").exists()


def test_evaluation_json_valid(cli_env, tmp_path, monkeypatch):
    out_dir = _run_cli(cli_env, tmp_path, monkeypatch)
    with open(out_dir / "evaluation.json") as f:
        data = json.load(f)
    assert "baseline" in data
    assert "overall" in data["baseline"]
    # All score values should be 0-100
    for k, v in data["baseline"].items():
        if v is not None:
            assert 0.0 <= v <= 100.0, f"Score {k}={v} out of range"


def test_cli_produces_score_progression(cli_env, tmp_path, monkeypatch):
    out_dir = _run_cli(cli_env, tmp_path, monkeypatch)
    assert (out_dir / "score_progression.png").exists()


def test_cli_produces_comparison(cli_env, tmp_path, monkeypatch):
    out_dir = _run_cli(cli_env, tmp_path, monkeypatch)
    assert (out_dir / "comparison.png").exists()


def test_cli_produces_feedback_json(cli_env, tmp_path, monkeypatch):
    out_dir = _run_cli(cli_env, tmp_path, monkeypatch)
    assert (out_dir / "feedback.json").exists()
