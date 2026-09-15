"""Extract the native 72-D libllsm2 coder representation from WAV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np

from .audio import AUDIO_SUFFIXES, read_audio
from .resample import resample_audio


def extract_f0_parselmouth(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int,
    frame_count: int,
    f0_min: float = 40.0,
    f0_max: float = 1_400.0,
) -> np.ndarray:
    """Extract a frame-aligned Praat autocorrelation F0 contour.

    Padding and truncation intentionally follow pyllsm2's integration tests so
    the F0 vector has exactly the number of frames expected by libllsm2.
    """

    try:
        import parselmouth
    except ImportError as error:
        raise RuntimeError(
            "F0 extraction needs praat-parselmouth; install "
            "vocoder/requirements-analysis.txt"
        ) from error

    left_padding = int(np.ceil(1.5 / f0_min * sample_rate))
    right_padding = (
        hop_length * ((len(waveform) - 1) // hop_length + 1)
        - len(waveform)
        + left_padding
        + 1
    )
    padded = np.pad(waveform, (left_padding, right_padding))
    pitch = parselmouth.Sound(padded, sample_rate).to_pitch_ac(
        time_step=hop_length / sample_rate,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    )
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float32)
    if f0.size < frame_count:
        f0 = np.pad(f0, (0, frame_count - f0.size))
    return np.ascontiguousarray(f0[:frame_count], dtype=np.float32)


def extract_f0_fcpe(
    waveform: np.ndarray,
    sample_rate: int,
    frame_count: int,
    f0_min: float,
    f0_max: float,
    device: str = "cpu",
    model=None,
) -> np.ndarray:
    """Extract singing F0 with the official torchfcpe inference package."""

    try:
        import torch
        from torchfcpe import spawn_bundled_infer_model
    except ImportError as error:
        raise RuntimeError(
            "FCPE extraction needs torch and torchfcpe; install "
            "vocoder/requirements-analysis.txt"
        ) from error
    audio = torch.from_numpy(waveform).float().view(1, -1, 1).to(device)
    model = model or spawn_bundled_infer_model(device=device)
    with torch.inference_mode():
        f0 = model.infer(
            audio,
            sr=sample_rate,
            decoder_mode="local_argmax",
            threshold=0.006,
            f0_min=f0_min,
            f0_max=f0_max,
            interp_uv=False,
            output_interp_target_length=frame_count,
        )
    values = f0.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
    return _fit_f0_length(values, frame_count)


def extract_f0_rmvpe(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int,
    frame_count: int,
    confidence_threshold: float = 0.03,
    model=None,
) -> np.ndarray:
    """Extract robust singing F0 using the CPU-friendly ONNX RMVPE package."""

    try:
        from rmvpe_onnx import RMVPE
    except ImportError as error:
        import sys

        version_note = (
            " RMVPE-ONNX requires Python 3.10 or newer."
            if sys.version_info < (3, 10)
            else ""
        )
        raise RuntimeError(
            "RMVPE extraction needs rmvpe-onnx; install "
            f"vocoder/requirements-analysis.txt.{version_note}"
        ) from error
    model = model or RMVPE()
    times, frequency, confidence, _ = model.predict(audio=waveform, sr=sample_rate)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    frequency = np.asarray(frequency, dtype=np.float32).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    target_times = np.arange(frame_count, dtype=np.float64) * hop_length / sample_rate
    if not times.size:
        return np.zeros(frame_count, dtype=np.float32)
    target_frequency = np.interp(target_times, times, frequency).astype(np.float32)
    target_confidence = np.interp(target_times, times, confidence)
    target_frequency[target_confidence < confidence_threshold] = 0.0
    return np.ascontiguousarray(target_frequency)


def _fit_f0_length(values: np.ndarray, frame_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < frame_count:
        values = np.pad(values, (0, frame_count - values.size))
    return np.ascontiguousarray(values[:frame_count], dtype=np.float32)


class F0Extractor:
    """Reusable F0 backend; heavy neural models are loaded at most once."""

    def __init__(self, backend: str = "fcpe", device: str = "cpu"):
        if backend not in {"fcpe", "rmvpe", "parselmouth"}:
            raise ValueError(f"unsupported F0 backend: {backend}")
        self.backend = backend
        self.device = device
        self._model = None

    def __call__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        hop_length: int,
        frame_count: int,
        f0_min: float,
        f0_max: float,
    ) -> np.ndarray:
        if self.backend == "fcpe":
            if self._model is None:
                try:
                    from torchfcpe import spawn_bundled_infer_model
                except ImportError as error:
                    raise RuntimeError("FCPE needs the 'analysis' dependency extra") from error
                self._model = spawn_bundled_infer_model(device=self.device)
            return extract_f0_fcpe(
                waveform,
                sample_rate,
                frame_count,
                f0_min,
                f0_max,
                device=self.device,
                model=self._model,
            )
        if self.backend == "rmvpe":
            if self._model is None:
                try:
                    from rmvpe_onnx import RMVPE
                except ImportError as error:
                    raise RuntimeError("RMVPE needs the 'rmvpe' dependency extra") from error
                self._model = RMVPE()
            values = extract_f0_rmvpe(
                waveform,
                sample_rate,
                hop_length,
                frame_count,
                model=self._model,
            )
            values[(values < f0_min) | (values > f0_max)] = 0.0
            return values
        return extract_f0_parselmouth(
            waveform, sample_rate, hop_length, frame_count, f0_min, f0_max
        )


def extract_f0(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int,
    frame_count: int,
    f0_min: float = 40.0,
    f0_max: float = 1_400.0,
    backend: str = "fcpe",
    device: str = "cpu",
) -> np.ndarray:
    """Dispatch to FCPE, RMVPE, or the Praat compatibility backend."""

    if backend == "fcpe":
        return extract_f0_fcpe(
            waveform, sample_rate, frame_count, f0_min, f0_max, device=device
        )
    if backend == "rmvpe":
        values = extract_f0_rmvpe(waveform, sample_rate, hop_length, frame_count)
        values[(values < f0_min) | (values > f0_max)] = 0.0
        return values
    if backend == "parselmouth":
        return extract_f0_parselmouth(
            waveform, sample_rate, hop_length, frame_count, f0_min, f0_max
        )
    raise ValueError(f"unsupported F0 backend: {backend}")


def analyze_waveform(
    waveform: np.ndarray,
    sample_rate: int = 48_000,
    hop_length: int = 256,
    f0_min: float = 40.0,
    f0_max: float = 1_400.0,
    nfft: int = 2_048,
    max_harmonics: int = 400,
    npsd: int = 128,
    f0_backend: str = "fcpe",
    f0_device: str = "cpu",
    f0_extractor: F0Extractor | None = None,
) -> np.ndarray:
    """Return native pyllsm2 coded features with shape ``[frames, 72]``."""

    try:
        import pyllsm2
    except ImportError as error:
        raise RuntimeError(
            "LLSM analysis needs pyllsm2; install "
            "vocoder/requirements-analysis.txt"
        ) from error

    waveform = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32).reshape(-1))
    if not waveform.size:
        raise ValueError("cannot analyze an empty waveform")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or infinity")
    frame_count = max(1, waveform.size // hop_length)
    if f0_extractor is None:
        f0 = extract_f0(
            waveform,
            sample_rate,
            hop_length,
            frame_count,
            f0_min=f0_min,
            f0_max=f0_max,
            backend=f0_backend,
            device=f0_device,
        )
    else:
        f0 = f0_extractor(
            waveform, sample_rate, hop_length, frame_count, f0_min, f0_max
        )

    options = pyllsm2.AnalysisOptions()
    options.thop = hop_length / float(sample_rate)
    options.maxnhar = max_harmonics
    options.maxnhar_e = 5
    options.npsd = npsd
    layer0 = None
    layer1 = None
    try:
        layer0 = pyllsm2.analyze(options, waveform, float(sample_rate), f0)
        layer1 = pyllsm2.to_layer1(layer0, nfft)
        with pyllsm2.Coder(layer1, order_spec=64, order_bap=5) as coder:
            # pyllsm2 0.2.0's convenience encode_frame asks an ordinary
            # calloc'ed coder vector for an internal length prefix and can
            # consequently report length zero. The native coder contract says
            # the length is exactly order_spec + order_bap + 3, so copy those
            # 72 values directly from the public raw API.
            rows = []
            for index in range(layer1.nfrm):
                encoded = pyllsm2.raw.lib.llsm_coder_encode(
                    coder.ptr, layer1.ptr.frames[index]
                )
                if encoded == pyllsm2.raw.ffi.NULL:
                    raise RuntimeError("llsm_coder_encode returned NULL")
                try:
                    row = np.frombuffer(
                        pyllsm2.raw.ffi.buffer(encoded, 72 * 4), dtype=np.float32
                    ).copy()
                finally:
                    pyllsm2.raw.free(encoded)
                rows.append(row)
            features = np.stack(rows).astype(np.float32, copy=False)
    finally:
        if layer1 is not None:
            layer1.close()
        if layer0 is not None:
            layer0.close()
        options.close()

    if features.shape != (frame_count, 72):
        raise RuntimeError(
            f"pyllsm2 returned {features.shape}; expected ({frame_count}, 72)"
        )
    if not np.isfinite(features).all():
        raise RuntimeError("pyllsm2 produced NaN or infinity")
    return np.ascontiguousarray(features)


def analyze_wav(path: str | Path, **kwargs) -> np.ndarray:
    sample_rate = int(kwargs.get("sample_rate", 48_000))
    waveform, native_sample_rate = read_audio(path)
    if native_sample_rate != sample_rate:
        waveform = resample_audio(waveform, native_sample_rate, sample_rate)
    return analyze_waveform(waveform, **kwargs)


def _jobs(source: Path, output: Path | None) -> Iterator[Tuple[Path, Path]]:
    if source.is_file():
        if source.suffix.lower() not in AUDIO_SUFFIXES:
            raise ValueError("input file must be WAV or FLAC")
        target = output if output is not None else source.with_suffix(".npy")
        if target.suffix.lower() != ".npy":
            target = target / source.with_suffix(".npy").name
        yield source, target
        return
    if not source.is_dir():
        raise FileNotFoundError(source)
    audio_paths = sorted(
        path for path in source.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES
    )
    for audio_path in audio_paths:
        relative = audio_path.relative_to(source).with_suffix(".npy")
        yield audio_path, (output / relative if output is not None else audio_path.with_suffix(".npy"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze WAV/FLAC into native [VUV,F0,Rd,64 spectrum,5 BAP] features"
    )
    parser.add_argument("input", type=Path, help="one WAV/FLAC or a directory searched recursively")
    parser.add_argument("--output", type=Path, help="output .npy or mirrored output directory")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f0-min", type=float, default=40.0)
    parser.add_argument("--f0-max", type=float, default=1_400.0)
    parser.add_argument(
        "--f0-backend",
        choices=("fcpe", "rmvpe", "parselmouth"),
        default="fcpe",
        help="FCPE is the singing-oriented default; RMVPE is the robust alternative",
    )
    parser.add_argument("--f0-device", default="cpu", help="FCPE device: cpu, cuda, or mps")
    parser.add_argument("--nfft", type=int, default=2_048)
    parser.add_argument("--max-harmonics", type=int, default=400)
    parser.add_argument("--npsd", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    jobs = list(_jobs(args.input, args.output))
    if not jobs:
        raise SystemExit(f"no WAV files found below {args.input}")
    f0_extractor = F0Extractor(args.f0_backend, args.f0_device)
    for index, (wav_path, feature_path) in enumerate(jobs, start=1):
        if feature_path.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing: {feature_path}")
            continue
        features = analyze_wav(
            wav_path,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            f0_min=args.f0_min,
            f0_max=args.f0_max,
            nfft=args.nfft,
            max_harmonics=args.max_harmonics,
            npsd=args.npsd,
            f0_backend=args.f0_backend,
            f0_device=args.f0_device,
            f0_extractor=f0_extractor,
        )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features, allow_pickle=False)
        print(f"[{index}/{len(jobs)}] {wav_path} -> {feature_path} {features.shape}")


if __name__ == "__main__":
    main()
