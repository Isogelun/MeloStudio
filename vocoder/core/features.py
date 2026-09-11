from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


FEATURE_NAMES: Tuple[str, ...] = (
    "VUV",
    "F0",
    "Rd",
    *(f"spectral_envelope_{index:02d}" for index in range(64)),
    *(f"BAP_{index}" for index in range(5)),
)


def default_feature_range() -> Tuple[Tensor, Tensor]:
    """Return conservative physical ranges for [VUV, F0, Rd, 64 SP, 5 BAP]."""

    minimum = torch.tensor([0.0, 0.0, 0.02] + [-100.0] * 64 + [0.0] * 5)
    maximum = torch.tensor([1.0, 2_000.0, 3.0] + [40.0] * 64 + [1.0] * 5)
    return minimum, maximum


class MaxMinNormalizer(nn.Module):
    """Map physical LLSM values to [-1, 1] without train-time statistics."""

    def __init__(self, minimum: Optional[Tensor] = None, maximum: Optional[Tensor] = None):
        super().__init__()
        if minimum is None or maximum is None:
            minimum, maximum = default_feature_range()
        if minimum.shape != (72,) or maximum.shape != (72,):
            raise ValueError("feature minimum and maximum must both have shape [72]")
        if torch.any(maximum <= minimum):
            raise ValueError("every feature maximum must be greater than its minimum")
        self.register_buffer("minimum", minimum.float().view(1, 72, 1))
        self.register_buffer("maximum", maximum.float().view(1, 72, 1))

    def forward(self, features: Tensor) -> Tensor:
        scaled = (features - self.minimum) / (self.maximum - self.minimum)
        return scaled.clamp(0.0, 1.0).mul(2.0).sub(1.0)

    def set_range(self, minimum: Tensor, maximum: Tensor) -> None:
        """Replace the range in-place so it remains part of the model checkpoint."""

        minimum = torch.as_tensor(minimum, dtype=self.minimum.dtype).reshape(1, 72, 1)
        maximum = torch.as_tensor(maximum, dtype=self.maximum.dtype).reshape(1, 72, 1)
        if torch.any(maximum <= minimum):
            raise ValueError("every feature maximum must be greater than its minimum")
        self.minimum.copy_(minimum.to(self.minimum.device))
        self.maximum.copy_(maximum.to(self.maximum.device))
