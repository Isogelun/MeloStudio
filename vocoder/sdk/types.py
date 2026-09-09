from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio import write_wav
from .errors import FeatureValueError, InvalidFeatureShape


@dataclass(frozen=True)
class LLSMFeatures:
    values: np.ndarray
    sample_rate: int | None = None
    hop_length: int | None = None
    source_sample_rate: int | None = None
    source_id: str | None = None
    feature_version: str = "llsm72-v1"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2 or 72 not in values.shape:
            raise InvalidFeatureShape(
                f"expected [T,72] or [72,T], got {tuple(values.shape)}"
            )
        if values.shape[0] == 72:
            values = values.T
        if values.shape[0] == 0:
            raise InvalidFeatureShape("features contain no frames")
        if not np.isfinite(values).all():
            raise FeatureValueError("features contain NaN or infinity")
        if self.feature_version != "llsm72-v1":
            raise FeatureValueError(f"unsupported feature version: {self.feature_version}")
        object.__setattr__(self, "values", np.ascontiguousarray(values))

    @classmethod
    def from_numpy(cls, values: np.ndarray, **metadata) -> "LLSMFeatures":
        return cls(values, **metadata)

    @property
    def frames(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class AudioResult:
    samples: np.ndarray
    sample_rate: int
    channels: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("audio result contains NaN or infinity")
        object.__setattr__(self, "samples", np.ascontiguousarray(values))

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        write_wav(output, self.samples, self.sample_rate)
        return output
