from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import F0Extractor
from .audio import AUDIO_SUFFIXES, read_audio
from .resample import resample_audio


def compare_contours(first: np.ndarray, second: np.ndarray) -> dict:
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    if first.shape != second.shape:
        raise ValueError("F0 contours must have equal lengths")
    first_voiced, second_voiced = first > 0, second > 0
    overlap = first_voiced & second_voiced
    cents = np.abs(1200 * np.log2(first[overlap] / second[overlap])) if overlap.any() else np.array([])
    return {
        "frames": len(first),
        "first_voiced_ratio": float(first_voiced.mean()) if len(first) else 0.0,
        "second_voiced_ratio": float(second_voiced.mean()) if len(second) else 0.0,
        "voicing_agreement": float((first_voiced == second_voiced).mean()) if len(first) else 0.0,
        "overlap_voiced_frames": int(overlap.sum()),
        "median_absolute_cents": float(np.median(cents)) if len(cents) else None,
        "p95_absolute_cents": float(np.percentile(cents, 95)) if len(cents) else None,
        "gross_pitch_error_ratio": float(np.mean(cents > 50)) if len(cents) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two F0 backends over an audio dataset")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first", choices=("fcpe", "rmvpe", "parselmouth"), default="fcpe")
    parser.add_argument("--second", choices=("fcpe", "rmvpe", "parselmouth"), default="rmvpe")
    parser.add_argument("--device", default="cpu", help="FCPE device")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f0-min", type=float, default=40.0)
    parser.add_argument("--f0-max", type=float, default=1_400.0)
    args = parser.parse_args()
    if args.first == args.second:
        raise SystemExit("choose two different F0 backends")
    paths = (
        [args.input]
        if args.input.is_file()
        else sorted(path for path in args.input.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    )
    if not paths:
        raise SystemExit("no WAV/FLAC files found")
    extractors = {
        args.first: F0Extractor(args.first, args.device),
        args.second: F0Extractor(args.second, args.device),
    }
    records = []
    for index, path in enumerate(paths, 1):
        try:
            waveform, native_rate = read_audio(path)
            if native_rate != args.sample_rate:
                waveform = resample_audio(waveform, native_rate, args.sample_rate)
            frames = max(1, len(waveform) // args.hop_length)
            contours = {
                name: extractor(
                    waveform, args.sample_rate, args.hop_length, frames,
                    args.f0_min, args.f0_max,
                )
                for name, extractor in extractors.items()
            }
            metrics = compare_contours(contours[args.first], contours[args.second])
            records.append({"file": str(path), "status": "ok", **metrics})
            print(f"[{index}/{len(paths)}] {path}: agreement={metrics['voicing_agreement']:.3f}")
        except Exception as error:
            records.append({"file": str(path), "status": "failed", "error": str(error)})
            print(f"[{index}/{len(paths)}] failed: {path}: {error}")
    valid = [row for row in records if row["status"] == "ok"]
    weighted_frames = sum(row["frames"] for row in valid)
    summary = {
        "files": len(records),
        "successful": len(valid),
        "failed": len(records) - len(valid),
        "frames": weighted_frames,
        "voicing_agreement": (
            sum(row["voicing_agreement"] * row["frames"] for row in valid) / weighted_frames
            if weighted_frames else None
        ),
        "median_file_absolute_cents": (
            float(np.median([row["median_absolute_cents"] for row in valid if row["median_absolute_cents"] is not None]))
            if any(row["median_absolute_cents"] is not None for row in valid) else None
        ),
    }
    report = {
        "config": vars(args) | {"input": str(args.input), "output": str(args.output)},
        "summary": summary,
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {args.output}")
    if summary["failed"]:
        raise SystemExit(f"completed with {summary['failed']} failed file(s)")


if __name__ == "__main__":
    main()
