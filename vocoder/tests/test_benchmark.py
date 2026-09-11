import numpy as np
import pytest

from vocoder.core.config import NHNVocoderConfig
from vocoder.core.dsp import DSPConfig, DSPControlPredictor
from vocoder.core.model import NHNVocoder
from vocoder.inference.benchmark import benchmark_checkpoint
from vocoder.training.checkpoint import load_checkpoint, save_checkpoint
from vocoder.training.post_train import save_post_training_checkpoint


def _checkpoint(tmp_path):
    config = NHNVocoderConfig(
        hop_length=32, residual_channels=16, skip_channels=16,
        primary_dilations=(1, 3), subbands=8, pqmf_taps=62,
        pqmf_cutoff_ratio=0.071, denoiser_channels=8,
        denoiser_dilations=(1, 3), max_harmonics=4,
    )
    path = tmp_path / "model.pt"
    save_checkpoint(path, NHNVocoder(config))
    return path


def test_benchmark_reports_float32_and_explicit_int8_status(tmp_path):
    features = np.zeros((4, 72), dtype=np.float32)
    features[:, 2] = 1
    report = benchmark_checkpoint(
        _checkpoint(tmp_path), features, dtypes=("float32", "int8"), warmup=0, runs=1
    )
    assert report["results"][0]["status"] == "ok"
    assert report["results"][0]["rtf"] >= 0
    assert report["results"][1]["status"] == "unsupported"


def test_benchmark_rejects_invalid_request(tmp_path):
    with pytest.raises(ValueError, match="runs"):
        benchmark_checkpoint(
            _checkpoint(tmp_path), np.zeros((4, 72), dtype=np.float32), runs=0
        )


def test_benchmark_includes_posttrained_dsp_head(tmp_path):
    base_path = _checkpoint(tmp_path)
    base = load_checkpoint(base_path)
    post = save_post_training_checkpoint(
        tmp_path / "post.pt",
        base,
        DSPControlPredictor(config=DSPConfig(hidden_channels=8)),
    )
    report = benchmark_checkpoint(
        post, np.zeros((4, 72), dtype=np.float32), warmup=0, runs=1
    )
    assert report["results"][0]["status"] == "ok"
    assert report["results"][0]["model_parameters_mb"] > 0
