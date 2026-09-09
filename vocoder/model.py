from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import NHNVocoderConfig
from .features import MaxMinNormalizer
from .layers import (
    FramewiseFIRFilter,
    MultibandNoiseSource,
    NonCausalWaveNet,
    PolyphaseFilterBank,
    SamePadConv1d,
)


class SpectrumGenerator(nn.Module):
    """Predict one frame-synchronous FIR kernel for each excitation branch."""

    def __init__(self, channels: int, hop_length: int, heads: int = 3):
        super().__init__()
        self.hop_length = hop_length
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Identity(),
                    nn.SiLU(),
                    SamePadConv1d(channels, channels, 3),
                    nn.SiLU(),
                    nn.Conv1d(channels, hop_length, 1),
                    nn.Identity(),
                )
                for _ in range(heads)
            ]
        )

    def forward(self, hidden: Tensor) -> Tuple[Tensor, ...]:
        return tuple(head(hidden) for head in self.heads)


class NoiseProcessor(nn.Module):
    def __init__(self, channels: int, bands: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, 64, 1), nn.SiLU(), nn.Conv1d(64, bands, 1)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return torch.sigmoid(self.layers(hidden))


class NHNVocoder(nn.Module):
    """Compact non-autoregressive neural harmonic-plus-noise vocoder.

    Input may be ``[batch, frames, 72]`` or ``[batch, 72, frames]``.  The
    returned waveform is ``[batch, 1, frames * hop_length]``.
    """

    def __init__(self, config: Optional[NHNVocoderConfig] = None):
        super().__init__()
        self.config = config or NHNVocoderConfig()
        cfg = self.config
        self.input_normalizer = MaxMinNormalizer()
        self.input_proj = nn.Conv1d(cfg.feature_dim, cfg.residual_channels, 1)
        self.primary_wn = NonCausalWaveNet(
            cfg.residual_channels,
            cfg.residual_channels,
            cfg.skip_channels,
            cfg.feature_dim,
            cfg.primary_dilations,
        )
        self.spectrum_gen = SpectrumGenerator(
            cfg.skip_channels, cfg.hop_length, cfg.spectrum_heads
        )
        self.fir_filters = nn.ModuleList(
            FramewiseFIRFilter(cfg.hop_length) for _ in range(cfg.spectrum_heads)
        )
        self.noise_proc = NoiseProcessor(cfg.skip_channels, cfg.noise_bands)
        self.noise_source = MultibandNoiseSource(cfg.noise_bands, cfg.sample_rate)
        self.filterbank = PolyphaseFilterBank(
            cfg.subbands, cfg.pqmf_taps, cfg.pqmf_cutoff_ratio, cfg.pqmf_beta
        )
        self.secondary_wn = NonCausalWaveNet(
            cfg.subbands,
            cfg.denoiser_channels,
            cfg.denoiser_channels,
            cfg.feature_dim,
            cfg.denoiser_dilations,
        )
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(cfg.denoiser_channels, cfg.denoiser_channels, 1),
            nn.ReLU(),
            nn.Conv1d(cfg.denoiser_channels, cfg.subbands, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # A freshly initialized model should be quiet rather than emit full-scale noise.
        final = self.output_proj[-1]
        if isinstance(final, nn.Conv1d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _canonicalize(self, features: Tensor) -> Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("features must have shape [T,72], [B,T,72], or [B,72,T]")
        dtype = self.input_proj.weight.dtype
        if features.shape[1] == self.config.feature_dim:
            return features.to(dtype=dtype)
        if features.shape[2] == self.config.feature_dim:
            return features.transpose(1, 2).to(dtype=dtype)
        raise ValueError(f"expected a 72-dimensional feature axis, got {tuple(features.shape)}")

    def _harmonic_source(self, physical: Tensor) -> Tensor:
        cfg = self.config
        frames = physical.shape[-1]
        samples = frames * cfg.hop_length
        f0 = F.interpolate(physical[:, 1:2], size=samples, mode="linear", align_corners=False)
        vuv = F.interpolate(physical[:, 0:1], size=samples, mode="linear", align_corners=False)
        f0 = f0.clamp(cfg.f0_min, cfg.f0_max)
        phase = torch.cumsum(f0 * (2.0 * math.pi / cfg.sample_rate), dim=-1)
        harmonics = torch.arange(
            1, cfg.max_harmonics + 1, device=physical.device, dtype=physical.dtype
        ).view(1, -1, 1)
        valid = (harmonics * f0 < cfg.sample_rate * 0.5).to(physical.dtype)
        source = (torch.sin(phase * harmonics) * valid / harmonics).sum(dim=1, keepdim=True)
        normalizer = (valid / harmonics).sum(dim=1, keepdim=True).clamp_min(1.0)
        return source / normalizer * vuv.clamp(0.0, 1.0)

    def _noise_source(
        self,
        physical: Tensor,
        learned_gains: Tensor,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        length = physical.shape[-1] * self.config.hop_length
        vuv = physical[:, 0:1].clamp(0.0, 1.0)
        bap = physical[:, 67:72].clamp(0.0, 1.0)
        # The five LLSM aperiodicity bands are smoothly expanded to the ten
        # synthesis filters; no information is collapsed to a global average.
        bap = F.interpolate(
            bap.transpose(1, 2), size=self.config.noise_bands,
            mode="linear", align_corners=False,
        ).transpose(1, 2)
        physical_gains = 0.15 + 0.85 * torch.maximum(1.0 - vuv, bap)
        return self.noise_source(learned_gains * physical_gains, length, generator)

    def forward(
        self,
        features: Tensor,
        generator: Optional[torch.Generator] = None,
        return_components: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        physical = self._canonicalize(features)
        condition = self.input_normalizer(physical)
        hidden = self.primary_wn(self.input_proj(condition), condition)
        heads = self.spectrum_gen(hidden)
        harmonic_excitation = self._harmonic_source(physical)
        noise_gains = self.noise_proc(hidden)
        noise_excitation = self._noise_source(physical, noise_gains, generator)
        energy = physical[:, 3:67].mean(dim=1, keepdim=True)
        energy = torch.cat(
            (torch.zeros_like(energy[..., :1]), energy[..., 1:] - energy[..., :-1]),
            dim=-1,
        ).abs()
        transient = F.interpolate(
            energy, size=harmonic_excitation.shape[-1], mode="nearest"
        ) * noise_excitation

        excitations = (harmonic_excitation, noise_excitation, transient)
        streams = [
            fir(excitations[index % len(excitations)], head)
            for index, (fir, head) in enumerate(zip(self.fir_filters, heads))
        ]
        coarse = torch.stack(streams, dim=0).sum(dim=0) / math.sqrt(len(streams))

        subbands, length = self.filterbank.analysis(coarse)
        denoised = self.secondary_wn(subbands, condition)
        residual = self.output_proj(denoised)
        waveform = torch.tanh(self.filterbank.synthesis(subbands + residual, length))
        if return_components:
            return waveform, {
                "coarse": coarse,
                "harmonic": streams[0],
                "noise": streams[1] if len(streams) > 1 else streams[0],
            }
        return waveform

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_export_mode(
        self, enabled: bool = True, *, onnx: bool = False
    ) -> "NHNVocoder":
        """Use an FFT-free FIR path compatible with graph exporters."""

        for module in self.fir_filters:
            module.export_mode = enabled
            module.onnx_mode = enabled and onnx
        return self

    @property
    def size_megabytes_fp32(self) -> float:
        return self.parameter_count * 4 / (1024**2)
