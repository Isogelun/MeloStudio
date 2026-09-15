from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from .errors import InvalidFeatureShape, UnsupportedInputType
from .types import LLSMFeatures


class FeatureAdapter(Protocol):
    def can_handle(self, source: object) -> bool: ...
    def convert(self, source: object) -> LLSMFeatures: ...


class FeatureAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: list[tuple[int, FeatureAdapter]] = []

    def register(self, adapter: FeatureAdapter, priority: int = 0) -> None:
        self._adapters.append((priority, adapter))
        self._adapters.sort(key=lambda item: item[0], reverse=True)

    def convert(self, source: object) -> LLSMFeatures:
        if isinstance(source, LLSMFeatures):
            return source
        for _, adapter in self._adapters:
            if adapter.can_handle(source):
                return adapter.convert(source)
        raise UnsupportedInputType(f"no feature adapter for {type(source).__name__}")


class ArrayAdapter:
    def can_handle(self, source: object) -> bool:
        return isinstance(source, (np.ndarray, torch.Tensor))

    def convert(self, source: object) -> LLSMFeatures:
        if isinstance(source, torch.Tensor):
            if source.ndim == 3 and source.shape[0] == 1:
                source = source.squeeze(0)
            source = source.detach().cpu().numpy()
        return LLSMFeatures.from_numpy(source)


class PathAdapter:
    def can_handle(self, source: object) -> bool:
        return isinstance(source, (str, Path)) and Path(source).suffix.lower() in {".npy", ".npz"}

    def convert(self, source: object) -> LLSMFeatures:
        path = Path(source)
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "features" in loaded.files:
                    values = loaded["features"]
                elif len(loaded.files) == 1:
                    values = loaded[loaded.files[0]]
                else:
                    raise InvalidFeatureShape("NPZ must contain a 'features' array")
            finally:
                loaded.close()
        else:
            values = loaded
        return LLSMFeatures.from_numpy(values, source_id=str(path))


class MappingAdapter:
    _required = {"vuv", "f0", "rd", "spectral_envelope", "bap"}

    def can_handle(self, source: object) -> bool:
        return isinstance(source, Mapping)

    def convert(self, source: object) -> LLSMFeatures:
        missing = self._required - set(source)
        if missing:
            raise InvalidFeatureShape(f"missing feature fields: {sorted(missing)}")
        columns = []
        for key, width in (("vuv", 1), ("f0", 1), ("rd", 1), ("spectral_envelope", 64), ("bap", 5)):
            value = np.asarray(source[key], dtype=np.float32)
            if width == 1:
                value = value.reshape(-1, 1)
            elif value.ndim != 2 or value.shape[-1] != width:
                raise InvalidFeatureShape(f"{key} must have shape [T,{width}]")
            columns.append(value)
        frames = {value.shape[0] for value in columns}
        if len(frames) != 1:
            raise InvalidFeatureShape("all feature fields must have the same frame count")
        return LLSMFeatures(
            np.concatenate(columns, axis=1),
            sample_rate=source.get("sample_rate"),
            hop_length=source.get("hop_length"),
            source_sample_rate=source.get("source_sample_rate"),
            source_id=source.get("source_id"),
        )


def default_registry() -> FeatureAdapterRegistry:
    registry = FeatureAdapterRegistry()
    registry.register(PathAdapter())
    registry.register(MappingAdapter())
    registry.register(ArrayAdapter())
    return registry
