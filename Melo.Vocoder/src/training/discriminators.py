from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _norm(module: nn.Module) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(module)


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN period discriminator, reduced for this compact vocoder."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        channels = (1, 16, 64, 128, 256)
        self.convs = nn.ModuleList(
            [
                _norm(nn.Conv2d(
                    channels[i], channels[i + 1], (5, 1),
                    stride=(3 if i < 3 else 1, 1), padding=(2, 0),
                ))
                for i in range(len(channels) - 1)
            ]
        )
        self.final = _norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, waveform: Tensor) -> Tuple[Tensor, list[Tensor]]:
        remainder = waveform.shape[-1] % self.period
        if remainder:
            waveform = F.pad(waveform, (0, self.period - remainder), mode="reflect")
        value = waveform.view(waveform.shape[0], 1, -1, self.period)
        maps = []
        for conv in self.convs:
            value = F.leaky_relu(conv(value), 0.1)
            maps.append(value)
        value = self.final(value)
        maps.append(value)
        return value.flatten(1), maps


class ScaleDiscriminator(nn.Module):
    """Grouped 1-D discriminator for waveform structure at one time scale."""

    def __init__(self):
        super().__init__()
        specs = (
            (1, 16, 15, 1, 7, 1),
            (16, 64, 41, 4, 20, 4),
            (64, 128, 41, 4, 20, 4),
            (128, 256, 41, 4, 20, 4),
            (256, 256, 5, 1, 2, 1),
        )
        self.convs = nn.ModuleList(
            [_norm(nn.Conv1d(a, b, k, stride=s, padding=p, groups=g)) for a, b, k, s, p, g in specs]
        )
        self.final = _norm(nn.Conv1d(256, 1, 3, padding=1))

    def forward(self, waveform: Tensor) -> Tuple[Tensor, list[Tensor]]:
        maps = []
        value = waveform
        for conv in self.convs:
            value = F.leaky_relu(conv(value), 0.1)
            maps.append(value)
        value = self.final(value)
        maps.append(value)
        return value.flatten(1), maps


class MultiPeriodScaleDiscriminator(nn.Module):
    """Training-only lightweight MPD + two-scale MSD ensemble."""

    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11), scales: int = 2):
        super().__init__()
        self.period_discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)
        self.scale_discriminators = nn.ModuleList(ScaleDiscriminator() for _ in range(scales))
        self.pool = nn.AvgPool1d(4, stride=2, padding=1)

    def forward(self, waveform: Tensor) -> list[Tuple[Tensor, list[Tensor]]]:
        results = [module(waveform) for module in self.period_discriminators]
        scaled = waveform
        for index, module in enumerate(self.scale_discriminators):
            if index:
                scaled = self.pool(scaled)
            results.append(module(scaled))
        return results


def discriminator_loss(real_outputs: Iterable, fake_outputs: Iterable) -> Tensor:
    losses = [
        (1.0 - real_score).square().mean() + fake_score.square().mean()
        for (real_score, _), (fake_score, _) in zip(real_outputs, fake_outputs)
    ]
    return torch.stack(losses).sum()


def generator_adversarial_loss(fake_outputs: Iterable) -> Tensor:
    return torch.stack([(1.0 - score).square().mean() for score, _ in fake_outputs]).sum()


def feature_matching_loss(real_outputs: Iterable, fake_outputs: Iterable) -> Tensor:
    losses = []
    for (_, real_maps), (_, fake_maps) in zip(real_outputs, fake_outputs):
        losses.extend(F.l1_loss(fake, real.detach()) for real, fake in zip(real_maps, fake_maps))
    return torch.stack(losses).mean()
