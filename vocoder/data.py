from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .audio import read_wav


class LLSMWavDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Paired ``name.npy`` (T x 72) and ``name.wav`` training data."""

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 48_000,
        hop_length: int = 256,
        segment_frames: Optional[int] = None,
        random_crop: bool = True,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.segment_frames = segment_frames
        self.random_crop = random_crop
        if segment_frames is not None and segment_frames <= 0:
            raise ValueError("segment_frames must be positive")
        self.items: List[Tuple[Path, Path]] = []
        for feature_path in sorted(self.root.rglob("*.npy")):
            wav_path = feature_path.with_suffix(".wav")
            if wav_path.is_file():
                self.items.append((feature_path, wav_path))
        if not self.items:
            raise ValueError(f"no matching .npy/.wav pairs found below {self.root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        feature_path, wav_path = self.items[index]
        features = np.load(feature_path, allow_pickle=False).astype(np.float32)
        if features.ndim != 2 or 72 not in features.shape:
            raise ValueError(f"{feature_path} must have shape [T,72] or [72,T]")
        if features.shape[0] == 72:
            features = features.T
        if not np.isfinite(features).all():
            raise ValueError(f"{feature_path} contains NaN or infinity")
        waveform, _ = read_wav(wav_path, self.sample_rate)
        expected = features.shape[0] * self.hop_length
        if waveform.shape[0] < expected:
            waveform = np.pad(waveform, (0, expected - waveform.shape[0]))
        waveform = waveform[:expected]

        if self.segment_frames is not None:
            segment_frames = self.segment_frames
            if features.shape[0] > segment_frames:
                maximum_start = features.shape[0] - segment_frames
                start = (
                    int(torch.randint(maximum_start + 1, ()).item())
                    if self.random_crop
                    else maximum_start // 2
                )
                features = features[start : start + segment_frames]
                sample_start = start * self.hop_length
                waveform = waveform[
                    sample_start : sample_start + segment_frames * self.hop_length
                ]
            elif features.shape[0] < segment_frames:
                frame_padding = segment_frames - features.shape[0]
                features = np.pad(features, ((0, frame_padding), (0, 0)))
                waveform = np.pad(
                    waveform, (0, frame_padding * self.hop_length)
                )
        return torch.from_numpy(features), torch.from_numpy(waveform).unsqueeze(0)
