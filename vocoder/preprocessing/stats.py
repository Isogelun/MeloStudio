from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_features(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False).astype(np.float64)
    if values.ndim != 2 or 72 not in values.shape:
        raise ValueError(f"{path} must have shape [T,72] or [72,T]")
    if values.shape[0] == 72:
        values = values.T
    if not values.shape[0]:
        raise ValueError(f"{path} contains no frames")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains NaN or infinity")
    return values


def compute_feature_stats(
    paths: Iterable[Path], sample_limit: int = 200_000
) -> dict[str, np.ndarray]:
    """Compute exact moments/ranges and sampled robust percentiles."""

    total = 0
    total_sum = np.zeros(72, dtype=np.float64)
    total_square = np.zeros(72, dtype=np.float64)
    minimum = np.full(72, np.inf, dtype=np.float64)
    maximum = np.full(72, -np.inf, dtype=np.float64)
    samples = []
    paths = list(paths)
    if not paths:
        raise ValueError("no .npy feature files found")
    per_file = max(1, sample_limit // len(paths))
    for path in paths:
        values = _load_features(path)
        total += values.shape[0]
        total_sum += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
        if values.shape[0] > per_file:
            indices = np.linspace(0, values.shape[0] - 1, per_file, dtype=np.int64)
            values = values[indices]
        samples.append(values)
    sampled = np.concatenate(samples, axis=0)
    mean = total_sum / total
    variance = np.maximum(total_square / total - np.square(mean), 0.0)
    p01, p99 = np.percentile(sampled, (1.0, 99.0), axis=0)
    # Constant or near-constant dimensions still need a valid normalization span.
    fallback = np.maximum(np.sqrt(variance), 1e-4)
    valid_span = p99 - p01 > 1e-6
    p01 = np.where(valid_span, p01, mean - fallback)
    p99 = np.where(valid_span, p99, mean + fallback)
    return {
        "count": np.asarray(total, dtype=np.int64),
        "mean": mean.astype(np.float32),
        "std": np.sqrt(variance).astype(np.float32),
        "minimum": minimum.astype(np.float32),
        "maximum": maximum.astype(np.float32),
        "p01": p01.astype(np.float32),
        "p99": p99.astype(np.float32),
    }


def load_feature_range(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stats:
        if "p01" not in stats or "p99" not in stats:
            raise ValueError("feature stats must contain p01 and p99")
        minimum = np.asarray(stats["p01"], dtype=np.float32)
        maximum = np.asarray(stats["p99"], dtype=np.float32)
    if minimum.shape != (72,) or maximum.shape != (72,):
        raise ValueError("feature stats p01/p99 must have shape [72]")
    return minimum, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NHN 72-D training feature statistics")
    parser.add_argument("data", type=Path, help="directory containing .npy features")
    parser.add_argument("output", type=Path, help="output feature_stats.npz")
    parser.add_argument("--sample-limit", type=int, default=200_000)
    args = parser.parse_args()
    paths = sorted(args.data.rglob("*.npy"))
    stats = compute_feature_stats(paths, sample_limit=args.sample_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **stats)
    print(
        f"wrote {args.output}: files={len(paths)} frames={int(stats['count'])} "
        f"sampled_percentiles<={args.sample_limit}"
    )


if __name__ == "__main__":
    main()
