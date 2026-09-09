from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024**2 if platform.system() == "Darwin" else value / 1024


def benchmark_checkpoint(
    checkpoint: str | Path,
    features: np.ndarray,
    *,
    device: str = "cpu",
    dtypes: tuple[str, ...] = ("float32",),
    warmup: int = 2,
    runs: int = 5,
    seed: int = 1234,
) -> dict:
    if features.ndim != 2 or features.shape[1] != 72 or features.shape[0] < 1:
        raise ValueError(f"features must have shape [T,72], got {features.shape}")
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    target_device = torch.device(device)
    results = []
    baseline = None
    for name in dtypes:
        if name == "int8":
            results.append(
                {
                    "dtype": name,
                    "status": "unsupported",
                    "reason": "the generator is Conv1d/FFT based; PyTorch dynamic int8 would not quantize it",
                }
            )
            continue
        if name not in DTYPES:
            raise ValueError(f"unsupported dtype: {name}")
        try:
            model = load_checkpoint(checkpoint, target_device).to(dtype=DTYPES[name])
            tensor = torch.from_numpy(features).unsqueeze(0).to(target_device, dtype=DTYPES[name])
            duration = features.shape[0] * model.config.hop_length / model.config.sample_rate
            if target_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target_device)
            rss_before = _peak_rss_mb()
            output = None
            times = []
            with torch.inference_mode():
                for index in range(warmup + runs):
                    generator = torch.Generator(device=target_device).manual_seed(seed)
                    _sync(target_device)
                    started = time.perf_counter()
                    output = model(tensor, generator=generator)
                    _sync(target_device)
                    elapsed = time.perf_counter() - started
                    if index >= warmup:
                        times.append(elapsed)
            values = output.float().cpu().numpy()
            if baseline is None and name == "float32":
                baseline = values
            results.append(
                {
                    "dtype": name,
                    "status": "ok",
                    "mean_seconds": float(np.mean(times)),
                    "p95_seconds": float(np.percentile(times, 95)),
                    "rtf": float(np.mean(times) / duration),
                    "audio_seconds": duration,
                    "peak_device_memory_mb": (
                        torch.cuda.max_memory_allocated(target_device) / 1024**2
                        if target_device.type == "cuda" else None
                    ),
                    "peak_process_rss_mb": _peak_rss_mb(),
                    "peak_process_rss_growth_mb": max(0.0, _peak_rss_mb() - rss_before),
                    "model_parameters_mb": model.size_megabytes_fp32,
                    "max_abs_difference_from_float32": (
                        float(np.max(np.abs(values - baseline))) if baseline is not None else None
                    ),
                }
            )
        except (RuntimeError, NotImplementedError) as error:
            results.append({"dtype": name, "status": "unsupported", "reason": str(error)})
    return {"device": str(target_device), "runs": runs, "warmup": warmup, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NHN inference RTF and device memory")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--features", type=Path)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--frames", type=int, help="synthetic feature frames")
    length.add_argument(
        "--seconds", type=float, default=2.0,
        help="synthetic audio duration; converted using checkpoint timing (default: 2)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtypes", default="float32,bfloat16,float16,int8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.features:
        features = np.load(args.features, allow_pickle=False).astype(np.float32)
        if features.shape[0] == 72:
            features = features.T
    else:
        if args.frames is not None:
            frames = args.frames
        else:
            timing_model = load_checkpoint(args.checkpoint, "cpu")
            frames = round(args.seconds * timing_model.config.sample_rate / timing_model.config.hop_length)
        if frames < 1:
            raise ValueError("benchmark duration must produce at least one feature frame")
        features = np.zeros((frames, 72), dtype=np.float32)
        features[:, 0] = 1
        features[:, 1] = 220
        features[:, 2] = 1
    report = benchmark_checkpoint(
        args.checkpoint,
        features,
        device=args.device,
        dtypes=tuple(item.strip() for item in args.dtypes.split(",") if item.strip()),
        warmup=args.warmup,
        runs=args.runs,
        seed=args.seed,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
