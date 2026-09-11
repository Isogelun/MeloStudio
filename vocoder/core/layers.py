from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SamePadConv1d(nn.Module):
    """Non-causal Conv1d whose output has the same temporal length as its input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.left = dilation * (kernel_size - 1) // 2
        self.right = dilation * (kernel_size - 1) - self.left
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(F.pad(value, (self.left, self.right)))


class GatedNonCausalConv1dBlock(nn.Module):
    """WaveNet-style tanh/sigmoid residual block with frame conditioning."""

    def __init__(
        self,
        in_channels: int,
        residual_channels: int,
        skip_channels: int,
        condition_channels: int,
        dilation: int,
    ):
        super().__init__()
        gate_channels = residual_channels * 2
        self.dilated_conv = SamePadConv1d(
            in_channels, gate_channels, kernel_size=3, dilation=dilation
        )
        self.cond_conv = nn.Conv1d(condition_channels, gate_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.input_residual = (
            nn.Identity()
            if in_channels == residual_channels
            else nn.Conv1d(in_channels, residual_channels, 1)
        )

    def forward(self, value: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor]:
        if condition.shape[-1] != value.shape[-1]:
            condition = F.interpolate(condition, size=value.shape[-1], mode="linear", align_corners=False)
        gates = self.dilated_conv(value) + self.cond_conv(condition)
        gate_a, gate_b = gates.chunk(2, dim=1)
        hidden = torch.tanh(gate_a) * torch.sigmoid(gate_b)
        residual = (self.input_residual(value) + self.res_conv(hidden)) * (2.0**-0.5)
        return residual, self.skip_conv(hidden)


class NonCausalWaveNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        residual_channels: int,
        skip_channels: int,
        condition_channels: int,
        dilations: Tuple[int, ...],
    ):
        super().__init__()
        blocks = []
        for index, dilation in enumerate(dilations):
            blocks.append(
                GatedNonCausalConv1dBlock(
                    in_channels if index == 0 else residual_channels,
                    residual_channels,
                    skip_channels,
                    condition_channels,
                    dilation,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        skip: Optional[Tensor] = None
        for block in self.blocks:
            value, current_skip = block(value, condition)
            skip = current_skip if skip is None else skip + current_skip
        if skip is None:
            raise RuntimeError("NonCausalWaveNet requires at least one block")
        return skip * (len(self.blocks) ** -0.5)


class PolyphaseFilterBank(nn.Module):
    """Cosine-modulated near-perfect-reconstruction PQMF filter bank.

    The previous implementation only de-interleaved samples into phases.  That is
    invertible, but aliases every phase channel.  This implementation applies a
    Kaiser-windowed prototype filter before decimation and the matching synthesis
    filters after interpolation.
    """

    def __init__(
        self,
        bands: int = 16,
        taps: int = 126,
        cutoff_ratio: float | None = None,
        beta: float = 9.0,
    ):
        super().__init__()
        if taps <= 0 or taps % 2:
            raise ValueError("taps must be a positive even number")
        self.bands = bands
        self.taps = taps
        cutoff_ratio = cutoff_ratio or 0.57 / bands
        positions = torch.arange(taps + 1, dtype=torch.float64) - taps / 2
        prototype = cutoff_ratio * torch.sinc(cutoff_ratio * positions)
        prototype = prototype * torch.kaiser_window(
            taps + 1, periodic=False, beta=beta, dtype=torch.float64
        )
        analysis = []
        synthesis = []
        for band in range(bands):
            carrier = (2 * band + 1) * math.pi / (2 * bands) * positions
            phase = ((-1) ** band) * math.pi / 4
            analysis.append(2 * prototype * torch.cos(carrier + phase))
            synthesis.append(2 * prototype * torch.cos(carrier - phase))
        self.register_buffer("analysis_filter", torch.stack(analysis).float().unsqueeze(1))
        self.register_buffer("synthesis_filter", torch.stack(synthesis).float().unsqueeze(0))

        updown = torch.zeros(bands, bands, bands)
        indices = torch.arange(bands)
        updown[indices, indices, 0] = 1.0
        self.register_buffer("updown_filter", updown)

    def analysis(self, waveform: Tensor) -> Tuple[Tensor, int]:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        original_length = waveform.shape[-1]
        padding = (-original_length) % self.bands
        waveform = F.pad(waveform, (0, padding))
        filtered = F.conv1d(
            F.pad(waveform, (self.taps // 2, self.taps // 2)), self.analysis_filter
        )
        subbands = F.conv1d(filtered, self.updown_filter, stride=self.bands)
        return subbands, original_length

    def synthesis(self, subbands: Tensor, length: int) -> Tensor:
        waveform = F.conv_transpose1d(
            subbands, self.updown_filter * self.bands, stride=self.bands
        )
        waveform = F.conv1d(
            F.pad(waveform, (self.taps // 2, self.taps // 2)), self.synthesis_filter
        )
        return waveform[..., :length]


class MultibandNoiseSource(nn.Module):
    """Differentiable ten-band aperiodic excitation with fixed FIR filters."""

    def __init__(self, bands: int, sample_rate: int, taps: int = 63):
        super().__init__()
        if taps % 2 == 0:
            raise ValueError("noise filter taps must be odd")
        nyquist = sample_rate / 2
        # Mel spacing spends more controls in the perceptually important low band.
        mel_max = 2595.0 * math.log10(1.0 + nyquist / 700.0)
        mel_edges = torch.linspace(0.0, mel_max, bands + 1, dtype=torch.float64)
        edges = 700.0 * (torch.pow(10.0, mel_edges / 2595.0) - 1.0) / sample_rate
        positions = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2
        window = torch.hann_window(taps, periodic=False, dtype=torch.float64)
        filters = []
        for low, high in zip(edges[:-1], edges[1:]):
            highpass = 2 * high * torch.sinc(2 * high * positions)
            lowpass = 2 * low * torch.sinc(2 * low * positions)
            kernel = (highpass - lowpass) * window
            kernel = kernel / kernel.square().sum().sqrt().clamp_min(1e-8)
            filters.append(kernel)
        self.register_buffer("filters", torch.stack(filters).float().unsqueeze(1))
        self.bands = bands
        self.padding = taps // 2

    def forward(
        self,
        gains: Tensor,
        length: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        gains = F.interpolate(gains, size=length, mode="linear", align_corners=False)
        noise = torch.randn(
            gains.shape[0], self.bands, length,
            device=gains.device, dtype=gains.dtype, generator=generator,
        )
        filtered = F.conv1d(noise, self.filters, padding=self.padding, groups=self.bands)
        return (filtered * gains).sum(dim=1, keepdim=True) / math.sqrt(self.bands)


class FramewiseFIRFilter(nn.Module):
    """Apply one predicted FIR kernel per non-overlapping excitation frame.

    Linear convolutions are evaluated with a 2*hop FFT and combined by
    overlap-add.  Coefficients are energy-scaled, which bounds the fresh
    model's output without removing the filter's learnable spectral gain.
    """

    def __init__(self, taps: int):
        super().__init__()
        self.taps = taps
        self.export_mode = False
        self.onnx_mode = False

    def forward(self, excitation: Tensor, coefficients: Tensor) -> Tensor:
        if excitation.ndim != 3 or excitation.shape[1] != 1:
            raise ValueError("excitation must have shape [B,1,samples]")
        if coefficients.ndim != 3 or coefficients.shape[1] != self.taps:
            raise ValueError(f"coefficients must have shape [B,{self.taps},frames]")
        batch, _, length = excitation.shape
        frames = coefficients.shape[-1]
        expected = frames * self.taps
        if length != expected:
            raise ValueError(f"expected {expected} excitation samples, got {length}")

        blocks = excitation.view(batch, frames, self.taps)
        kernels = coefficients.transpose(1, 2)
        kernels = torch.tanh(kernels) / math.sqrt(self.taps)
        if self.onnx_mode:
            if batch != 1:
                raise ValueError("ONNX FIR export currently supports batch size 1")
            filtered = torch.stack(
                [
                    F.conv1d(
                        blocks[:, index : index + 1],
                        kernels[:, index : index + 1].flip(-1),
                        padding=self.taps - 1,
                    ).squeeze(1)
                    for index in range(frames)
                ],
                dim=1,
            )
            filtered = F.pad(filtered, (0, 1))
        elif self.export_mode:
            padded = F.pad(blocks, (self.taps - 1, self.taps - 1))
            windows = padded.unfold(-1, self.taps, 1)
            filtered = (windows * kernels.flip(-1).unsqueeze(-2)).sum(-1)
            filtered = F.pad(filtered, (0, 1))
        else:
            fft_size = self.taps * 2
            filtered = torch.fft.irfft(
                torch.fft.rfft(blocks, n=fft_size)
                * torch.fft.rfft(kernels, n=fft_size),
                n=fft_size,
            )
        leading, trailing = filtered[..., : self.taps], filtered[..., self.taps :]
        # The last tail lies beyond the requested signal.  Every other tail is
        # added to the next block, producing differentiable partitioned OLA.
        output = torch.cat(
            (leading[:, :1], leading[:, 1:] + trailing[:, :-1]), dim=1
        )
        return output.reshape(batch, 1, length)
