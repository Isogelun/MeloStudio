import pytest

torch = pytest.importorskip("torch")

from vocoder import NHNVocoder, NHNVocoderConfig
from vocoder.layers import FramewiseFIRFilter, PolyphaseFilterBank


def test_pqmf_filterbank_has_near_perfect_reconstruction():
    filterbank = PolyphaseFilterBank(8, taps=126, cutoff_ratio=0.071)
    waveform = torch.randn(2, 1, 4096)
    subbands, length = filterbank.analysis(waveform)
    assert subbands.shape == (2, 8, 512)
    reconstructed = filterbank.synthesis(subbands, length)
    interior = (..., slice(256, -256))
    relative_error = (waveform[interior] - reconstructed[interior]).square().mean().sqrt() / waveform[interior].square().mean().sqrt()
    assert float(relative_error) < 0.03


def test_framewise_fir_overlap_add_has_gradients():
    fir = FramewiseFIRFilter(32)
    excitation = torch.randn(2, 1, 96, requires_grad=True)
    coefficients = torch.randn(2, 32, 3, requires_grad=True)
    waveform = fir(excitation, coefficients)
    assert waveform.shape == excitation.shape
    waveform.square().mean().backward()
    assert torch.isfinite(excitation.grad).all()
    assert torch.isfinite(coefficients.grad).all()


def test_output_length_and_layouts():
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
    )
    model = NHNVocoder(config).eval()
    features = torch.zeros(2, 7, 72)
    features[..., 0] = 1.0
    features[..., 1] = 220.0
    features[..., 2] = 1.0
    with torch.inference_mode():
        waveform_a = model(features, generator=torch.Generator().manual_seed(7))
        waveform_b = model(features.transpose(1, 2), generator=torch.Generator().manual_seed(7))
    assert waveform_a.shape == (2, 1, 7 * 32)
    torch.testing.assert_close(waveform_a, waveform_b)
    assert torch.isfinite(waveform_a).all()
    assert waveform_a.abs().max() <= 1.0


def test_rejects_wrong_feature_dimension():
    model = NHNVocoder()
    with pytest.raises(ValueError, match="72-dimensional"):
        model(torch.zeros(1, 10, 71))
