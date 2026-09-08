from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


def resample_audio(
    waveform: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
    *,
    target_length: int | None = None,
) -> np.ndarray:
    """Deterministic band-limited polyphase resampling with exact output length."""

    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample rates must be positive")
    values = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32).reshape(-1))
    if not np.isfinite(values).all():
        raise ValueError("waveform contains NaN or infinity")
    if source_sample_rate == target_sample_rate:
        output = values.copy()
    else:
        divisor = math.gcd(source_sample_rate, target_sample_rate)
        output = resample_poly(
            values,
            target_sample_rate // divisor,
            source_sample_rate // divisor,
            window=("kaiser", 8.6),
            padtype="constant",
        ).astype(np.float32, copy=False)
    expected = (
        target_length
        if target_length is not None
        else round(len(values) * target_sample_rate / source_sample_rate)
    )
    if expected < 0:
        raise ValueError("target_length must be non-negative")
    if len(output) < expected:
        output = np.pad(output, (0, expected - len(output)))
    return np.ascontiguousarray(output[:expected], dtype=np.float32)


def bandwidth_degrade(
    waveform_48k: np.ndarray,
    source_sample_rate: int,
    analysis_sample_rate: int = 48_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an aligned low-rate source and its 48 kHz analysis waveform."""

    if source_sample_rate > analysis_sample_rate:
        raise ValueError("source sample rate may not exceed analysis sample rate")
    original = np.asarray(waveform_48k, dtype=np.float32).reshape(-1)
    source_length = round(len(original) * source_sample_rate / analysis_sample_rate)
    low_rate = resample_audio(
        original, analysis_sample_rate, source_sample_rate, target_length=source_length
    )
    analysis = resample_audio(
        low_rate, source_sample_rate, analysis_sample_rate, target_length=len(original)
    )
    return low_rate, analysis


def measure_roundtrip_delay(
    source_sample_rate: int, analysis_sample_rate: int = 48_000
) -> int:
    """Measure peak displacement of the exact offline down/up sampling path."""

    length = 4096
    center = length // 2
    impulse = np.zeros(length, dtype=np.float32)
    impulse[center] = 1.0
    _, restored = bandwidth_degrade(impulse, source_sample_rate, analysis_sample_rate)
    return int(np.argmax(np.abs(restored)) - center)
