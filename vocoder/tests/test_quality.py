import torch

from vocoder.discriminators import (
    MultiPeriodScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from vocoder.losses import NHNVocoderLoss


def test_composite_quality_loss_is_finite_and_differentiable():
    estimate = torch.randn(1, 1, 4096, requires_grad=True)
    target = torch.randn_like(estimate)
    features = torch.zeros(1, 16, 72)
    features[..., 0] = 1.0
    features[..., 1] = 220.0
    criterion = NHNVocoderLoss(resolutions=((1024, 256, 1024), (512, 128, 512)))
    loss, components = criterion(estimate, target, features, return_components=True)
    assert set(components) == {
        "waveform", "spectral", "mel", "phase",
        "instantaneous_frequency", "harmonic", "total",
    }
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(estimate.grad).all()


def test_lightweight_mpd_msd_losses():
    discriminator = MultiPeriodScaleDiscriminator()
    real = torch.randn(1, 1, 2048)
    fake = torch.randn(1, 1, 2048, requires_grad=True)
    real_outputs = discriminator(real)
    fake_outputs = discriminator(fake)
    assert len(real_outputs) == 7
    loss = (
        discriminator_loss(real_outputs, fake_outputs)
        + generator_adversarial_loss(fake_outputs)
        + feature_matching_loss(real_outputs, fake_outputs)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(fake.grad).all()
