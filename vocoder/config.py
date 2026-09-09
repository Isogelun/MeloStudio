from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple


@dataclass
class NHNVocoderConfig:
    """Configuration for the 72-dimensional LLSM NHN vocoder.

    The defaults follow the supplied model summary and result in roughly 12 MB
    of float32 parameters.  ``hop_length`` is also the number of samples emitted
    by one LLSM frame.
    """

    sample_rate: int = 48_000
    hop_length: int = 256
    feature_dim: int = 72
    feature_schema: str = "llsm72-v1"
    task_mode: str = "vocoder"
    trained_input_sample_rates: Tuple[int, ...] = (48_000,)
    residual_channels: int = 256
    skip_channels: int = 256
    primary_dilations: Tuple[int, ...] = (1, 3, 9)
    spectrum_heads: int = 3
    noise_bands: int = 10
    subbands: int = 16
    pqmf_taps: int = 254
    pqmf_cutoff_ratio: float = 0.0355
    pqmf_beta: float = 9.0
    denoiser_channels: int = 64
    denoiser_dilations: Tuple[int, ...] = (1, 3, 9, 27, 81)
    max_harmonics: int = 64
    f0_min: float = 20.0
    f0_max: float = 2_000.0

    def __post_init__(self) -> None:
        if self.feature_dim != 72:
            raise ValueError("NHN expects exactly 72 LLSM features")
        if self.feature_schema != "llsm72-v1":
            raise ValueError(f"unsupported feature schema: {self.feature_schema}")
        if self.task_mode not in {"vocoder", "vocoder_bwe"}:
            raise ValueError(f"unsupported task mode: {self.task_mode}")
        if not self.trained_input_sample_rates or any(
            rate <= 0 or rate > self.sample_rate for rate in self.trained_input_sample_rates
        ):
            raise ValueError("trained input sample rates must be within model sample rate")
        if self.hop_length % self.subbands:
            raise ValueError("hop_length must be divisible by subbands")
        if min(self.sample_rate, self.hop_length, self.subbands) <= 0:
            raise ValueError("sample_rate, hop_length and subbands must be positive")
        if self.pqmf_taps <= 0 or self.pqmf_taps % 2:
            raise ValueError("pqmf_taps must be a positive even number")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["primary_dilations"] = list(self.primary_dilations)
        value["denoiser_dilations"] = list(self.denoiser_dilations)
        value["trained_input_sample_rates"] = list(self.trained_input_sample_rates)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "NHNVocoderConfig":
        allowed = {item.name for item in fields(cls)}
        data = {key: item for key, item in value.items() if key in allowed}
        for key in ("primary_dilations", "denoiser_dilations", "trained_input_sample_rates"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)
