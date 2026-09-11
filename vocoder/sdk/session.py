from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..analyze import F0Extractor, analyze_waveform
from ..audio import AUDIO_SUFFIXES, read_audio
from ..checkpoint import load_checkpoint
from ..dsp import DSPConfig, DSPControlPredictor, DifferentiableDSP
from ..model import NHNVocoder
from ..resample import resample_audio
from .adapters import FeatureAdapterRegistry, default_registry
from .errors import FeatureConfigMismatch, FeatureValueError, UnsupportedInputType
from .types import AudioResult, DSPControl, LLSMFeatures, SynthesisRequest


class VocoderSession:
    def __init__(
        self,
        model: NHNVocoder,
        *,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        adapters: FeatureAdapterRegistry | None = None,
        strict_bandwidth: bool = False,
        dsp_predictor: DSPControlPredictor | None = None,
        dsp_config: DSPConfig | None = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.seed = seed
        self.adapters = adapters or default_registry()
        self.strict_bandwidth = strict_bandwidth
        self.dsp = DifferentiableDSP(dsp_config).to(self.device).eval()
        self.dsp_predictor = (
            dsp_predictor.to(self.device).eval() if dsp_predictor is not None else None
        )
        self._f0_extractors: dict[tuple[str, str], F0Extractor] = {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs) -> "VocoderSession":
        device = kwargs.get("device", "cpu")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        dsp_config = DSPConfig.from_dict(payload.get("dsp_config"))
        predictor = None
        if "dsp_predictor" in payload:
            predictor = DSPControlPredictor(config=dsp_config)
            predictor.load_state_dict(payload["dsp_predictor"])
        return cls(
            load_checkpoint(path, device),
            dsp_predictor=predictor,
            dsp_config=dsp_config,
            **kwargs,
        )

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

    @staticmethod
    def _control_array(
        controls: DSPControl | np.ndarray,
        frames: int,
    ) -> np.ndarray:
        values = (
            controls.to_array(frames)
            if isinstance(controls, DSPControl)
            else np.asarray(controls, dtype=np.float32)
        )
        if values.shape == (6, frames):
            values = values.T
        if values.shape != (frames, 6) or not np.isfinite(values).all():
            raise FeatureConfigMismatch(
                f"DSP controls must have finite shape [{frames},6], got {values.shape}"
            )
        lower = np.array([-18.0, -1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        upper = np.array([18.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        if np.any(values < lower) or np.any(values > upper):
            raise FeatureValueError("DSP control is outside its documented range")
        return np.ascontiguousarray(values)

    def _control_tensor(
        self,
        controls: DSPControl | np.ndarray,
        frames: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = self._control_array(controls, frames)
        return torch.from_numpy(np.ascontiguousarray(values.T)).unsqueeze(0).to(
            self.device, dtype=dtype
        )

    def synthesize(
        self,
        source: object,
        *,
        seed: int | None = None,
        dsp_controls: DSPControl | np.ndarray | None = None,
    ) -> AudioResult:
        features = self.adapters.convert(source)
        self._validate_metadata(features)
        tensor = torch.from_numpy(features.values).unsqueeze(0).to(self.device)
        generator = self._generator(seed)
        control_source = "bypass"
        with torch.inference_mode():
            waveform = self.model(tensor, generator=generator)
            if dsp_controls is not None:
                controls = self._control_tensor(
                    dsp_controls, features.frames, waveform.dtype
                )
                waveform = self.dsp(waveform, controls, generator=generator)
                control_source = "explicit"
            elif self.dsp_predictor is not None:
                controls = self.dsp_predictor(tensor.to(self.dsp_predictor.network[0].weight.dtype))
                waveform = self.dsp(waveform, controls, generator=generator)
                control_source = "predicted"
            waveform = waveform[0, 0].float().cpu().numpy()
        return AudioResult(
            waveform,
            self.sample_rate,
            metadata={
                "frames": features.frames,
                "hop_length": self.model.config.hop_length,
                "task_mode": self.model.config.task_mode,
                "source_sample_rate": features.source_sample_rate,
                "warnings": list(features.warnings),
                "dsp_control_source": control_source,
            },
        )

    def synthesize_request(
        self,
        request: SynthesisRequest,
        *,
        chunk_frames: int | None = None,
        overlap_frames: int = 32,
        seed: int | None = None,
    ) -> AudioResult:
        result = (
            self.synthesize_chunked(
                request.features,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
                seed=seed,
                dsp_controls=request.dsp,
            )
            if chunk_frames is not None
            else self.synthesize(request.features, seed=seed, dsp_controls=request.dsp)
        )
        metadata = dict(result.metadata)
        metadata["request_metadata"] = dict(request.metadata)
        return AudioResult(result.samples, result.sample_rate, metadata=metadata)

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
        dsp_controls: DSPControl | np.ndarray | None = None,
    ) -> AudioResult:
        features = self.adapters.convert(source)
        self._validate_metadata(features)
        if chunk_frames <= overlap_frames or overlap_frames < 1:
            raise ValueError("chunk_frames must be greater than positive overlap_frames")
        if features.frames <= chunk_frames:
            return self.synthesize(features, seed=seed, dsp_controls=dsp_controls)
        full_controls = (
            self._control_array(dsp_controls, features.frames)
            if dsp_controls is not None
            else None
        )
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
            chunk_controls = (
                np.asarray(full_controls)[start:end]
                if full_controls is not None
                else None
            )
            audio = self.synthesize(
                chunk,
                seed=base_seed + index,
                dsp_controls=chunk_controls,
            ).samples.astype(np.float64)
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
                "dsp_control_source": (
                    "explicit" if dsp_controls is not None else
                    "predicted" if self.dsp_predictor is not None else "bypass"
                ),
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
