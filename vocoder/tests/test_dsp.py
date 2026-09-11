import sys

import numpy as np
import pytest
import torch

from vocoder.checkpoint import save_checkpoint
from vocoder.audio import write_wav
from vocoder.config import NHNVocoderConfig
from vocoder.dsp import DSPConfig, DSPControlPredictor, DifferentiableDSP
from vocoder.model import NHNVocoder
from vocoder.post_train import save_post_training_checkpoint
from vocoder.sdk import DSPControl, FeatureValueError, SynthesisRequest, VocoderSession
from vocoder.train import main as train_main


def _config():
    return NHNVocoderConfig(
        hop_length=32,
        residual_channels=16,
        skip_channels=16,
        primary_dilations=(1, 3),
        subbands=8,
        pqmf_taps=62,
        pqmf_cutoff_ratio=0.071,
        denoiser_channels=8,
        denoiser_dilations=(1, 3),
        max_harmonics=4,
    )


def _features(frames=8):
    values = np.zeros((frames, 72), dtype=np.float32)
    values[:, 0] = 1
    values[:, 1] = 220
    values[:, 2] = 1
    return values


def test_zero_dsp_controls_are_exact_bypass_and_trainable():
    waveform = torch.randn(2, 1, 256).tanh().requires_grad_()
    controls = torch.zeros(2, 6, 8, requires_grad=True)
    output = DifferentiableDSP()(waveform, controls)
    torch.testing.assert_close(output, waveform)
    output.square().mean().backward()
    assert controls.grad is not None
    assert torch.isfinite(controls.grad).all()


def test_sdk_request_applies_explicit_controls(tmp_path):
    checkpoint = tmp_path / "base.pt"
    save_checkpoint(checkpoint, NHNVocoder(_config()))
    session = VocoderSession.from_checkpoint(checkpoint)
    dry = session.synthesize(_features(), seed=4)
    request = SynthesisRequest(
        _features(),
        DSPControl(gain_db=-6.0, harmonic_tilt=0.25, limiter_amount=0.4),
        metadata={"speaker": "demo"},
    )
    wet = session.synthesize_request(request, seed=4)
    assert wet.metadata["dsp_control_source"] == "explicit"
    assert wet.metadata["request_metadata"] == {"speaker": "demo"}
    assert len(wet.samples) == len(dry.samples)
    assert not np.array_equal(wet.samples, dry.samples)


def test_dsp_control_validation():
    with pytest.raises(FeatureValueError):
        DSPControl(gain_db=30.0).to_array(4)
    with pytest.raises(FeatureValueError):
        DSPControl.from_mapping({"unknown": 1.0})


def test_posttrained_checkpoint_automatically_loads_predictor(tmp_path):
    base = NHNVocoder(_config())
    predictor = DSPControlPredictor(config=DSPConfig(hidden_channels=8))
    checkpoint = save_post_training_checkpoint(
        tmp_path / "post.pt", base, predictor
    )
    session = VocoderSession.from_checkpoint(checkpoint)
    result = session.synthesize(_features(), seed=9)
    assert result.metadata["dsp_control_source"] == "predicted"
    assert np.isfinite(result.samples).all()


def test_frozen_base_post_training_only_gradients_predictor():
    base = NHNVocoder(_config()).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    predictor = DSPControlPredictor(config=DSPConfig(hidden_channels=8))
    dsp = DifferentiableDSP(predictor.config)
    features = torch.from_numpy(_features()).unsqueeze(0)
    with torch.no_grad():
        dry = base(features, generator=torch.Generator().manual_seed(3))
    controls = predictor(features)
    wet = dsp(dry, controls, generator=torch.Generator().manual_seed(4))
    wet.square().mean().backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in predictor.parameters())


def test_post_training_cli_writes_loadable_checkpoint(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    frames = 64
    np.save(data / "sample.npy", _features(frames))
    write_wav(data / "sample.wav", np.zeros(frames * 32, dtype=np.float32), 48_000)
    base_path = tmp_path / "base.pt"
    save_checkpoint(base_path, NHNVocoder(_config()))
    output = tmp_path / "post"
    config = tmp_path / "post.yaml"
    config.write_text(
        f"""
config_type: nhn_dsp_post_train
description: Exercise unified DSP post-training dispatch.
common:
  device: cpu
  amp: false
post_train:
  data: {data.name}
  base_checkpoint: {base_path.name}
  output: {output.name}
  epochs: 1
  batch_size: 1
  segment_seconds: 0.0426667
  log_every: 1
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "nhn-train", "--config", str(config),
    ])
    train_main()
    checkpoint = output / "latest.pt"
    assert checkpoint.is_file()
    assert VocoderSession.from_checkpoint(checkpoint).dsp_predictor is not None
