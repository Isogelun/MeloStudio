from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..analyze import F0Extractor, analyze_waveform
from ..audio import AUDIO_SUFFIXES, read_audio
from ..checkpoint import load_checkpoint
from ..model import NHNVocoder
from ..resample import resample_audio
from .adapters import FeatureAdapterRegistry, default_registry
from .errors import FeatureConfigMismatch, UnsupportedInputType
from .types import AudioResult, LLSMFeatures


class VocoderSession:
    def __init__(
        self,
        model: NHNVocoder,
        *,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        adapters: FeatureAdapterRegistry | None = None,
        strict_bandwidth: bool = False,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.seed = seed
        self.adapters = adapters or default_registry()
        self.strict_bandwidth = strict_bandwidth
        self._f0_extractors: dict[tuple[str, str], F0Extractor] = {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> "VocoderSession":
        device = kwargs.get("device", "cpu")
        return cls(load_checkpoint(path, device), **kwargs)

    @property
    def sample_rate(self) -> int:
        return self.model.config.sample_rate

    def _validate_metadata(self, features: LLSMFeatures) -> None:
        cfg = self.model.config
        if features.sample_rate is not None and features.sample_rate != cfg.sample_rate:
            raise FeatureConfigMismatch(
                f"feature analysis rate is {features.sample_rate}, checkpoint expects {cfg.sample_rate}"
            )
        if features.hop_length is not None and features.hop_length != cfg.hop_length:
            raise FeatureConfigMismatch(
                f"feature hop is {features.hop_length}, checkpoint expects {cfg.hop_length}"
            )

    def _generator(self, seed: int | None) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(self.seed if seed is None else seed)

    def synthesize(self, source: object, *, seed: int | None = None) -> AudioResult:
        features = self.adapters.convert(source)
        self._validate_metadata(features)
        tensor = torch.from_numpy(features.values).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            waveform = self.model(tensor, generator=self._generator(seed))[0, 0].cpu().numpy()
        return AudioResult(
            waveform,
            self.sample_rate,
            metadata={
                "frames": features.frames,
                "hop_length": self.model.config.hop_length,
                "task_mode": self.model.config.task_mode,
                "source_sample_rate": features.source_sample_rate,
                "warnings": list(features.warnings),
            },
        )

    def synthesize_batch(
        self, sources: Iterable[object], *, seeds: Iterable[int] | None = None
    ) -> list[AudioResult]:
        values = list(sources)
        seed_values = list(seeds) if seeds is not None else [self.seed + index for index in range(len(values))]
        if len(seed_values) != len(values):
            raise ValueError("seeds and sources must have equal lengths")
        return [self.synthesize(source, seed=seed) for source, seed in zip(values, seed_values)]

    def synthesize_chunked(
        self,
        source: object,
        *,
        chunk_frames: int = 750,
        overlap_frames: int = 32,
        seed: int | None = None,
    ) -> AudioResult:
        features = self.adapters.convert(source)
        self._validate_metadata(features)
        if chunk_frames <= overlap_frames or overlap_frames < 1:
            raise ValueError("chunk_frames must be greater than positive overlap_frames")
        if features.frames <= chunk_frames:
            return self.synthesize(features, seed=seed)
        hop = self.model.config.hop_length
        length = features.frames * hop
        accumulated = np.zeros(length, dtype=np.float64)
        weights = np.zeros(length, dtype=np.float64)
        stride = chunk_frames - overlap_frames
        base_seed = self.seed if seed is None else seed
        starts = list(range(0, features.frames, stride))
        for index, start in enumerate(starts):
            end = min(features.frames, start + chunk_frames)
            chunk = LLSMFeatures(
                features.values[start:end], sample_rate=features.sample_rate,
                hop_length=features.hop_length, source_sample_rate=features.source_sample_rate,
                source_id=features.source_id, feature_version=features.feature_version,
                warnings=features.warnings,
            )
            audio = self.synthesize(chunk, seed=base_seed + index).samples.astype(np.float64)
            window = np.ones(len(audio), dtype=np.float64)
            fade = min(overlap_frames * hop, len(audio) // 2)
            if start > 0:
                window[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
            if end < features.frames:
                window[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False)
            sample_start, sample_end = start * hop, end * hop
            accumulated[sample_start:sample_end] += audio * window
            weights[sample_start:sample_end] += window
            if end == features.frames:
                break
        output = (accumulated / np.maximum(weights, 1e-12)).astype(np.float32)
        return AudioResult(
            output, self.sample_rate,
            metadata={
                "frames": features.frames,
                "hop_length": hop,
                "task_mode": self.model.config.task_mode,
                "source_sample_rate": features.source_sample_rate,
                "warnings": list(features.warnings),
                "chunked": True,
                "chunk_frames": chunk_frames,
                "overlap_frames": overlap_frames,
            },
        )

    def synthesize_waveform(
        self,
        source: str | Path | np.ndarray,
        input_sample_rate: int | None = None,
        *,
        f0_backend: str = "fcpe",
        f0_device: str = "cpu",
        chunk_frames: int | None = None,
        overlap_frames: int = 32,
        seed: int | None = None,
    ) -> AudioResult:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                raise UnsupportedInputType("waveform path must be WAV or FLAC")
            waveform, native_rate = read_audio(path)
        else:
            if input_sample_rate is None:
                raise ValueError("input_sample_rate is required for waveform arrays")
            waveform, native_rate = np.asarray(source, dtype=np.float32), input_sample_rate
        if native_rate > self.sample_rate:
            raise FeatureConfigMismatch(
                f"input rate {native_rate} exceeds fixed output rate {self.sample_rate}"
            )
        source_warnings = []
        cfg = self.model.config
        if native_rate < self.sample_rate:
            if cfg.task_mode != "vocoder_bwe":
                message = "checkpoint was not trained for bandwidth extension"
                if self.strict_bandwidth:
                    raise FeatureConfigMismatch(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                source_warnings.append(message)
            if native_rate not in cfg.trained_input_sample_rates:
                message = f"input rate {native_rate} was not listed in checkpoint training rates"
                if self.strict_bandwidth:
                    raise FeatureConfigMismatch(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                source_warnings.append(message)
            waveform = resample_audio(waveform, native_rate, self.sample_rate)
        key = (f0_backend, f0_device)
        if key not in self._f0_extractors:
            self._f0_extractors[key] = F0Extractor(f0_backend, f0_device)
        extractor = self._f0_extractors[key]
        values = analyze_waveform(
            waveform,
            sample_rate=self.sample_rate,
            hop_length=cfg.hop_length,
            f0_min=max(40.0, cfg.f0_min),
            f0_max=min(1_400.0, cfg.f0_max),
            f0_backend=f0_backend,
            f0_device=f0_device,
            f0_extractor=extractor,
        )
        features = LLSMFeatures(
            values,
            sample_rate=self.sample_rate,
            hop_length=cfg.hop_length,
            source_sample_rate=native_rate,
            warnings=tuple(source_warnings),
        )
        return (
            self.synthesize_chunked(
                features,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
                seed=seed,
            )
            if chunk_frames is not None
            else self.synthesize(features, seed=seed)
        )
