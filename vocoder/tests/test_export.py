import pytest
import torch

from vocoder.checkpoint import load_checkpoint, save_checkpoint
from vocoder.config import NHNVocoderConfig
from vocoder.export import export_inference_checkpoint, export_onnx, export_torchscript
from vocoder.model import NHNVocoder


def _checkpoint(tmp_path):
    config = NHNVocoderConfig(
        hop_length=32, residual_channels=16, skip_channels=16,
        primary_dilations=(1, 3), subbands=8, pqmf_taps=62,
        pqmf_cutoff_ratio=0.071, denoiser_channels=8,
        denoiser_dilations=(1, 3), max_harmonics=4,
    )
    path = tmp_path / "training.pt"
    model = NHNVocoder(config)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(path, model, optimizer=optimizer, extra={"discriminator": {"unused": 1}})
    return path


def test_pure_checkpoint_drops_training_state(tmp_path):
    source = _checkpoint(tmp_path)
    output = export_inference_checkpoint(source, tmp_path / "inference.pt")
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["checkpoint_type"] == "inference"
    assert "optimizer" not in payload
    assert "discriminator" not in payload
    assert load_checkpoint(output).config.sample_rate == 48_000


def test_torchscript_export_runs(tmp_path):
    source = _checkpoint(tmp_path)
    output = export_torchscript(source, tmp_path / "model.ts", frames=7)
    model = torch.jit.load(str(output))
    waveform = model(torch.zeros(1, 7, 72))
    assert waveform.shape == (1, 1, 7 * 32)
    assert model(torch.zeros(1, 9, 72)).shape == (1, 1, 9 * 32)


def test_fixed_shape_onnx_export_runs(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    source = _checkpoint(tmp_path)
    output = export_onnx(source, tmp_path / "model.onnx", frames=7)
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    waveform = session.run(None, {"features": torch.zeros(1, 7, 72).numpy()})[0]
    assert waveform.shape == (1, 1, 7 * 32)
