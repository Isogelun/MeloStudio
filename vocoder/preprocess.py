from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .analyze import F0Extractor, analyze_wav
from .stats import compute_feature_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare paired WAV/72-D LLSM data and feature statistics in one run"
    )
    parser.add_argument("input", type=Path, help="raw WAV directory")
    parser.add_argument("output", type=Path, help="training dataset output directory")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--f0-min", type=float, default=40.0)
    parser.add_argument("--f0-max", type=float, default=1_400.0)
    parser.add_argument(
        "--f0-backend", choices=("fcpe", "rmvpe", "parselmouth"), default="fcpe"
    )
    parser.add_argument("--f0-device", default="cpu")
    parser.add_argument("--nfft", type=int, default=2_048)
    parser.add_argument("--max-harmonics", type=int, default=400)
    parser.add_argument("--npsd", type=int, default=128)
    parser.add_argument("--stats-name", default="feature_stats.npz")
    parser.add_argument("--sample-limit", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"input directory does not exist: {source_root}")
    if output_root != source_root and output_root.is_relative_to(source_root):
        raise SystemExit("output may not be nested inside input; it would be scanned again")
    wav_paths = sorted(source_root.rglob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"no WAV files found below {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    extractor = F0Extractor(args.f0_backend, args.f0_device)
    successful = []
    records = []
    for index, source_wav in enumerate(wav_paths, start=1):
        relative = source_wav.relative_to(source_root)
        target_wav = output_root / relative
        target_feature = target_wav.with_suffix(".npy")
        try:
            if target_feature.exists() and not args.overwrite:
                if target_wav != source_wav and not target_wav.exists():
                    target_wav.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_wav, target_wav)
                features = np.load(target_feature, mmap_mode="r", allow_pickle=False)
                if features.ndim != 2 or 72 not in features.shape:
                    raise ValueError(f"existing feature has invalid shape {features.shape}")
                frame_count = int(features.shape[1] if features.shape[0] == 72 else features.shape[0])
                status = "skipped"
            else:
                features = analyze_wav(
                    source_wav,
                    sample_rate=args.sample_rate,
                    hop_length=args.hop_length,
                    f0_min=args.f0_min,
                    f0_max=args.f0_max,
                    nfft=args.nfft,
                    max_harmonics=args.max_harmonics,
                    npsd=args.npsd,
                    f0_backend=args.f0_backend,
                    f0_device=args.f0_device,
                    f0_extractor=extractor,
                )
                target_wav.parent.mkdir(parents=True, exist_ok=True)
                if target_wav != source_wav:
                    shutil.copy2(source_wav, target_wav)
                np.save(target_feature, features, allow_pickle=False)
                frame_count = int(features.shape[0])
                status = "processed"
            successful.append(target_feature)
            records.append(
                {
                    "file": str(relative),
                    "status": status,
                    "frames": frame_count,
                    "seconds": frame_count * args.hop_length / args.sample_rate,
                }
            )
            print(f"[{index}/{len(wav_paths)}] {status}: {relative} -> {frame_count} frames")
        except Exception as error:
            records.append(
                {"file": str(relative), "status": "failed", "error": str(error)}
            )
            print(f"[{index}/{len(wav_paths)}] failed: {relative}: {error}")
            if args.fail_fast:
                raise

    if not successful:
        raise SystemExit("preprocessing failed: no usable feature files")
    stats = compute_feature_stats(successful, sample_limit=args.sample_limit)
    stats_path = output_root / args.stats_name
    np.savez(stats_path, **stats)
    report = {
        "config": {
            "sample_rate": args.sample_rate,
            "hop_length": args.hop_length,
            "f0_backend": args.f0_backend,
            "f0_min": args.f0_min,
            "f0_max": args.f0_max,
            "nfft": args.nfft,
            "max_harmonics": args.max_harmonics,
            "npsd": args.npsd,
        },
        "summary": {
            "total": len(records),
            "processed": sum(item["status"] == "processed" for item in records),
            "skipped": sum(item["status"] == "skipped" for item in records),
            "failed": sum(item["status"] == "failed" for item in records),
            "frames": int(stats["count"]),
            "hours": float(stats["count"]) * args.hop_length / args.sample_rate / 3600,
        },
        "files": records,
    }
    report_path = output_root / "preprocess_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"statistics: {stats_path}")
    print(f"report: {report_path}")
    if report["summary"]["failed"]:
        raise SystemExit(f"completed with {report['summary']['failed']} failed file(s)")


if __name__ == "__main__":
    main()
