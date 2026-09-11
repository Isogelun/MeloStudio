from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DSP_CONTROL_NAMES = (
    "gain_db",
    "harmonic_tilt",
    "breathiness",
    "transient_gain",
    "deesser_amount",
    "limiter_amount",
)


@dataclass(frozen=True)
class DSPConfig:
    control_dim: int = len(DSP_CONTROL_NAMES)
    hidden_channels: int = 64
    noise_level: float = 0.025
    tilt_strength: float = 0.35
    transient_strength: float = 0.25
    deesser_kernel: int = 9

    def __post_init__(self) -> None:
        if self.control_dim != len(DSP_CONTROL_NAMES):
            raise ValueError(f"DSP control_dim must be {len(DSP_CONTROL_NAMES)}")
        if self.hidden_channels <= 0 or self.noise_level < 0:
            raise ValueError("invalid DSP configuration")
        if self.deesser_kernel < 3 or self.deesser_kernel % 2 == 0:
            raise ValueError("deesser_kernel must be an odd number >= 3")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None) -> "DSPConfig":
        return cls(**(value or {}))


class DifferentiableDSP(nn.Module):
    """Small, bypassable waveform DSP controlled at the LLSM frame rate.

    Controls use physical ranges documented by ``DSPControl``. A zero control
    matrix is an exact bypass, which makes the module safe to add after an
    already-trained base vocoder.
    """

    def __init__(self, config: DSPConfig | None = None):
        super().__init__()
        self.config = config or DSPConfig()

    def forward(
        self,
        waveform: Tensor,
        controls: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape [B,1,samples]")
        if controls.ndim != 3 or controls.shape[1] != self.config.control_dim:
            raise ValueError(
                f"controls must have shape [B,{self.config.control_dim},frames]"
            )
        if controls.shape[0] != waveform.shape[0]:
            raise ValueError("waveform and controls batch sizes differ")

        control = F.interpolate(
            controls, size=waveform.shape[-1], mode="linear", align_corners=False
        )
        gain_db = control[:, 0:1].clamp(-18.0, 18.0)
        tilt = control[:, 1:2].clamp(-1.0, 1.0)
        breath = control[:, 2:3].clamp(0.0, 1.0)
        transient = control[:, 3:4].clamp(-1.0, 1.0)
        deesser = control[:, 4:5].clamp(0.0, 1.0)
        limiter = control[:, 5:6].clamp(0.0, 1.0)

        output = waveform * torch.pow(10.0, gain_db / 20.0)
        previous = F.pad(output[..., :-1], (1, 0))
        difference = output - previous
        output = output + self.config.tilt_strength * tilt * difference

        local_energy = F.avg_pool1d(
            output.square(), kernel_size=33, stride=1, padding=16
        ).clamp_min(1e-8).sqrt()
        noise = torch.randn(
            output.shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )
        output = output + breath * self.config.noise_level * local_energy * noise

        previous = F.pad(output[..., :-1], (1, 0))
        output = output + self.config.transient_strength * transient * (output - previous)

        smooth = F.avg_pool1d(
            output,
            kernel_size=self.config.deesser_kernel,
            stride=1,
            padding=self.config.deesser_kernel // 2,
        )
        output = output - deesser * (output - smooth) * 0.75

        drive = 1.0 + 4.0 * limiter
        limited = torch.tanh(output * drive) / torch.tanh(drive)
        output = output * (1.0 - limiter) + limited * limiter
        return output.clamp(-1.0, 1.0)


class DSPControlPredictor(nn.Module):
    """Predict frame-rate DSP controls while leaving the base vocoder frozen."""

    def __init__(self, feature_dim: int = 72, config: DSPConfig | None = None):
        super().__init__()
        self.config = config or DSPConfig()
        hidden = self.config.hidden_channels
        self.network = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, 1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, self.config.control_dim, 1),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        # Positive-only controls start close to bypass but retain gradients.
        with torch.no_grad():
            final.bias[2] = -6.0
            final.bias[4] = -6.0
            final.bias[5] = -6.0

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,T,72] or [B,72,T]")
        if features.shape[1] != 72:
            if features.shape[2] != 72:
                raise ValueError("DSP predictor expects a 72-dimensional feature axis")
            features = features.transpose(1, 2)
        raw = self.network(features)
        signed = torch.tanh(raw[:, (0, 1, 3)])
        positive = torch.sigmoid(raw[:, (2, 4, 5)])
        return torch.stack(
            (
                signed[:, 0] * 12.0,
                signed[:, 1],
                positive[:, 0],
                signed[:, 2],
                positive[:, 1],
                positive[:, 2],
            ),
            dim=1,
        )


class DSPAugmentedVocoder(nn.Module):
    """Deployment wrapper combining a base vocoder and learned DSP control head."""

    def __init__(
        self,
        base: nn.Module,
        predictor: DSPControlPredictor,
        dsp: DifferentiableDSP | None = None,
    ):
        super().__init__()
        self.base = base
        self.predictor = predictor
        self.dsp = dsp or DifferentiableDSP(predictor.config)
        self.config = base.config

    def forward(
        self, features: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        waveform = self.base(features, generator=generator)
        controls = self.predictor(features)
        return self.dsp(waveform, controls, generator=generator)

    @property
    def size_megabytes_fp32(self) -> float:
        return sum(parameter.numel() for parameter in self.parameters()) * 4 / (1024**2)
