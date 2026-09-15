from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


AUDIO_SUFFIXES = {".wav", ".flac"}


def read_audio(
    path: str | Path, expected_sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Read PCM/float WAV or FLAC as finite mono float32 in [-1, 1]."""

    path = Path(path)
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"unsupported audio extension: {path.suffix}")
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError(f"audio is empty: {path}")
    waveform = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError(f"audio contains NaN or infinity: {path}")
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(f"expected {expected_sample_rate} Hz audio, got {sample_rate} Hz")
    return waveform, int(sample_rate)


def read_wav(
    path: str | Path, expected_sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Backward-compatible alias; now also supports FLAC and non-16-bit WAV."""

    return read_audio(path, expected_sample_rate)


def write_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("cannot write waveform containing NaN or infinity")
    sf.write(path, np.clip(values, -1.0, 1.0), sample_rate, subtype="PCM_16")
