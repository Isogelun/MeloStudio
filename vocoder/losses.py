from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mel_filter(sample_rate: int, fft_size: int, bands: int) -> Tensor:
    hz = torch.linspace(0, sample_rate / 2, fft_size // 2 + 1)
    mel_max = 2595.0 * torch.log10(torch.tensor(1.0 + sample_rate / 1400.0))
    mel_edges = torch.linspace(0.0, mel_max, bands + 2)
    edges = 700.0 * (torch.pow(10.0, mel_edges / 2595.0) - 1.0)
    lower = (hz[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    upper = (edges[2:, None] - hz[None]) / (edges[2:] - edges[1:-1])[:, None]
    return torch.minimum(lower, upper).clamp_min(0.0)


class NHNVocoderLoss(nn.Module):
    """Spectral, Mel, circular phase/IF and F0-harmonic reconstruction loss."""

    def __init__(
        self,
        sample_rate: int = 48_000,
        resolutions: Sequence[Tuple[int, int, int]] = (
            (1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)
        ),
        waveform_weight: float = 0.1,
        mel_weight: float = 1.0,
        phase_weight: float = 0.05,
        instantaneous_frequency_weight: float = 0.1,
        harmonic_weight: float = 0.25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.resolutions = tuple(resolutions)
        self.waveform_weight = waveform_weight
        self.mel_weight = mel_weight
        self.phase_weight = phase_weight
        self.instantaneous_frequency_weight = instantaneous_frequency_weight
        self.harmonic_weight = harmonic_weight
        self.register_buffer("mel_filter", _mel_filter(sample_rate, 1024, 80))

    @staticmethod
    def _stft(value: Tensor, fft_size: int, hop: int, window_length: int) -> Tensor:
        window = torch.hann_window(window_length, device=value.device, dtype=value.dtype)
        return torch.stft(value, fft_size, hop, window_length, window, return_complex=True)

    def _harmonic_loss(self, est: Tensor, ref: Tensor, features: Tensor | None) -> Tensor:
        if features is None:
            return est.real.new_zeros(())
        f0, vuv = (
            (features[:, 1], features[:, 0])
            if features.shape[1] == 72
            else (features[..., 1], features[..., 0])
        )
        frames = est.shape[-1]
        f0 = F.interpolate(f0.unsqueeze(1), size=frames, mode="linear", align_corners=False).squeeze(1)
        vuv = F.interpolate(vuv.unsqueeze(1), size=frames, mode="nearest").squeeze(1)
        harmonics = torch.arange(1, 33, device=est.device).view(1, -1, 1)
        bins = (harmonics * f0[:, None] * 1024 / self.sample_rate).round().long()
        valid = (bins < est.shape[-2]) & (vuv[:, None] > 0.5)
        bins = bins.clamp(0, est.shape[-2] - 1)
        est_h = est.abs().gather(1, bins).clamp_min(1e-7).log()
        ref_h = ref.abs().gather(1, bins).clamp_min(1e-7).log()
        return ((est_h - ref_h).abs() * valid).sum() / valid.sum().clamp_min(1)

    def forward(
        self,
        estimate: Tensor,
        target: Tensor,
        features: Tensor | None = None,
        return_components: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        estimate, target = estimate.squeeze(1), target.squeeze(1)
        components: Dict[str, Tensor] = {"waveform": F.l1_loss(estimate, target)}
        spectral = phase = inst_freq = estimate.new_zeros(())
        harmonic = mel_loss = estimate.new_zeros(())
        for index, (fft_size, hop, window_length) in enumerate(self.resolutions):
            est = self._stft(estimate, fft_size, hop, window_length)
            ref = self._stft(target, fft_size, hop, window_length)
            est_mag, ref_mag = est.abs().clamp_min(1e-7), ref.abs().clamp_min(1e-7)
            spectral = spectral + torch.linalg.vector_norm(ref_mag - est_mag) / torch.linalg.vector_norm(ref_mag).clamp_min(1e-7)
            spectral = spectral + F.l1_loss(est_mag.log(), ref_mag.log())
            weight = (ref_mag / ref_mag.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-7)).clamp_max(4.0)
            phase = phase + ((1.0 - torch.cos(torch.angle(est) - torch.angle(ref))) * weight).mean()
            est_if = torch.angle(est[..., 1:] * est[..., :-1].conj())
            ref_if = torch.angle(ref[..., 1:] * ref[..., :-1].conj())
            inst_freq = inst_freq + (1.0 - torch.cos(est_if - ref_if)).mean()
            if index == 0:
                est_mel = torch.einsum("mf,bft->bmt", self.mel_filter, est_mag)
                ref_mel = torch.einsum("mf,bft->bmt", self.mel_filter, ref_mag)
                mel_loss = F.l1_loss(est_mel.clamp_min(1e-5).log(), ref_mel.clamp_min(1e-5).log())
                harmonic = self._harmonic_loss(est, ref, features)
        count = len(self.resolutions)
        components.update(
            spectral=spectral / (2 * count), mel=mel_loss, phase=phase / count,
            instantaneous_frequency=inst_freq / count, harmonic=harmonic,
        )
        total = (
            self.waveform_weight * components["waveform"] + components["spectral"]
            + self.mel_weight * components["mel"] + self.phase_weight * components["phase"]
            + self.instantaneous_frequency_weight * components["instantaneous_frequency"]
            + self.harmonic_weight * components["harmonic"]
        )
        components["total"] = total
        return (total, components) if return_components else total


MultiResolutionSTFTLoss = NHNVocoderLoss
