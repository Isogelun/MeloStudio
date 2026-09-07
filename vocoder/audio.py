from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path, expected_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if width != 2:
            raise ValueError(f"only 16-bit PCM WAV is supported, got {width * 8}-bit")
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768.0
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(f"expected {expected_sample_rate} Hz WAV, got {sample_rate} Hz")
    return samples, sample_rate


def write_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

