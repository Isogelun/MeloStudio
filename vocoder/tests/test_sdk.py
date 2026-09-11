import numpy as np
import pytest
import torch

from vocoder.core.config import NHNVocoderConfig
from vocoder.core.model import NHNVocoder
from vocoder.training.checkpoint import save_checkpoint
from vocoder.sdk import DSPControl, FeatureConfigMismatch, LLSMFeatures, VocoderSession


def _session(tmp_path):
    config = NHNVocoderConfig(
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
        task_mode="vocoder_bwe",
        trained_input_sample_rates=(16_000, 48_000),
    )
    path = tmp_path / "model.pt"
    save_checkpoint(path, NHNVocoder(config))
    return VocoderSession.from_checkpoint(path)


def _features(frames=7):
    values = np.zeros((frames, 72), dtype=np.float32)
    values[:, 0] = 1
    values[:, 1] = 220
    values[:, 2] = 1
    return values


def test_sdk_array_mapping_and_seed_are_consistent(tmp_path):
    session = _session(tmp_path)
    values = _features()
    first = session.synthesize(values, seed=7)
    second = session.synthesize(LLSMFeatures(values), seed=7)
    np.testing.assert_array_equal(first.samples, second.samples)
    mapping = {
        "vuv": values[:, 0], "f0": values[:, 1], "rd": values[:, 2],
        "spectral_envelope": values[:, 3:67], "bap": values[:, 67:72],
    }
    third = session.synthesize(mapping, seed=7)
    np.testing.assert_array_equal(first.samples, third.samples)
    assert first.sample_rate == 48_000
    assert len(first.samples) == 7 * 32


def test_sdk_chunked_length_and_batch(tmp_path):
    session = _session(tmp_path)
    values = LLSMFeatures(
        _features(12), source_sample_rate=16_000, warnings=("test warning",)
    )
    chunked = session.synthesize_chunked(values, chunk_frames=7, overlap_frames=2)
    assert len(chunked.samples) == 12 * 32
    assert np.isfinite(chunked.samples).all()
    assert chunked.metadata["source_sample_rate"] == 16_000
    assert chunked.metadata["warnings"] == ["test warning"]
    assert len(session.synthesize_batch([values, values])) == 2


def test_sdk_rejects_feature_timing_mismatch(tmp_path):
    session = _session(tmp_path)
    with pytest.raises(FeatureConfigMismatch):
        session.synthesize(LLSMFeatures(_features(), sample_rate=44_100))


def test_chunked_explicit_dsp_controls_keep_length(tmp_path):
    session = _session(tmp_path)
    result = session.synthesize_chunked(
        _features(12),
        chunk_frames=7,
        overlap_frames=2,
        dsp_controls=DSPControl(gain_db=np.linspace(-3, 3, 12)),
    )
    assert len(result.samples) == 12 * 32
    assert result.metadata["dsp_control_source"] == "explicit"
