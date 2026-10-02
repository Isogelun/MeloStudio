from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze import F0Extractor, analyze_waveform
from .audio import AUDIO_SUFFIXES, read_audio, write_wav
from .quality import analyze_quality
from .resample import bandwidth_degrade, measure_roundtrip_delay, resample_audio
from .stats import compute_feature_stats


def parse_sample_rates(value: str) -> tuple[int, ...]:
    try:
        rates = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("sample rates must be comma-separated integers") from error
    if not rates or any(rate < 8_000 or rate > 48_000 for rate in rates):
        raise argparse.ArgumentTypeError("sample rates must be within 8000..48000")
    return rates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build aligned multi-rate LLSM conditions with 48 kHz waveform targets"
    )
    parser.add_argument("input", type=Path, help="directory of 48 kHz WAV/FLAC masters")
    parser.add_argument("output", type=Path, help="BWE dataset output directory")
    parser.add_argument(
        "--source-sample-rates",
        type=parse_sample_rates,
        default=parse_sample_rates("8000,12000,16000,22050,24000,32000,44100,48000"),
    )
    parser.add_argument("--target-sample-rate", type=int, default=48_000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--f0-min", type=float, default=40.0)
    parser.add_argument("--f0-max", type=float, default=1_400.0)
    parser.add_argument(
        "--f0-backend", choices=("fcpe", "rmvpe", "parselmouth"), default="fcpe"
    )
    parser.add_argument("--f0-device", default="cpu")
    parser.add_argument("--nfft", type=int, default=2_048)
    parser.add_argument("--max-harmonics", type=int, default=400)
    parser.add_argument("--npsd", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=200_000)
    parser.add_argument("--allow-target-resample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _relative_id(path: Path) -> str:
    return path.with_suffix("").as_posix()


def _split_ids(ids: list[str], validation_split: float, seed: int) -> set[str]:
    if len(ids) < 2 or validation_split <= 0:
        return set()
    count = min(len(ids) - 1, max(1, round(len(ids) * validation_split)))
    ranked = sorted(
        ids,
        key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).digest(),
    )
    return set(ranked[:count])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    source_root, output_root = args.input.resolve(), args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"input directory does not exist: {source_root}")
    if args.target_sample_rate != 48_000:
        raise SystemExit("this model family currently requires a 48000 Hz target")
    if not 0 <= args.validation_split < 1:
        raise SystemExit("validation-split must be in [0,1)")
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise SystemExit("BWE output must be outside the input master directory")

    sources = sorted(
        path for path in source_root.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not sources:
        raise SystemExit(f"no WAV/FLAC files found below {source_root}")
    ids = [_relative_id(path.relative_to(source_root)) for path in sources]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate relative stems found (for example both name.wav and name.flac)")

    extractor = F0Extractor(args.f0_backend, args.f0_device)
    feature_paths: list[Path] = []
    manifest_rows: list[dict] = []
    file_reports: list[dict] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for file_index, source_path in enumerate(sources, 1):
        relative = source_path.relative_to(source_root)
        item_id = _relative_id(relative)
        try:
            target, native_rate = read_audio(source_path)
            target_warnings = []
            if native_rate != args.target_sample_rate:
                if not args.allow_target_resample:
                    raise ValueError(
                        f"master must be 48000 Hz, got {native_rate}; "
                        "use --allow-target-resample only when this is intentional"
                    )
                target = resample_audio(target, native_rate, args.target_sample_rate)
                target_warnings.append("target_resampled")
            frame_count = len(target) // args.hop_length
            if frame_count < 1:
                raise ValueError("audio is shorter than one model frame")
            target = np.ascontiguousarray(target[: frame_count * args.hop_length])
            target_relative = Path("targets") / relative.with_suffix(".wav")
            target_path = output_root / target_relative
            if args.overwrite or not target_path.exists():
                write_wav(target_path, target, args.target_sample_rate)

            variant_reports = []
            for rate in args.source_sample_rates:
                feature_relative = (
                    Path("features") / relative.parent / f"{relative.stem}@{rate}.npy"
                )
                feature_path = output_root / feature_relative
                try:
                    if feature_path.exists() and not args.overwrite:
                        features = np.load(feature_path, allow_pickle=False)
                        if features.shape != (frame_count, 72):
                            raise ValueError(
                                f"existing feature shape {features.shape}, expected ({frame_count},72)"
                            )
                        low_rate, analysis_waveform = bandwidth_degrade(
                            target, rate, args.target_sample_rate
                        )
                        status = "skipped"
                    else:
                        low_rate, analysis_waveform = bandwidth_degrade(
                            target, rate, args.target_sample_rate
                        )
                        features = analyze_waveform(
                            analysis_waveform,
                            sample_rate=args.target_sample_rate,
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
                        if features.shape != (frame_count, 72):
                            raise RuntimeError(
                                f"feature shape {features.shape}, expected ({frame_count},72)"
                            )
                        feature_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(feature_path, features, allow_pickle=False)
                        status = "processed"

                    quality = analyze_quality(low_rate, features)
                    expected_source_samples = round(
                        len(target) * rate / args.target_sample_rate
                    )
                    row = {
                        "id": item_id,
                        "features": feature_relative.as_posix(),
                        "target": target_relative.as_posix(),
                        "frames": frame_count,
                        "target_samples": len(target),
                        "source_samples": len(low_rate),
                        "source_sample_rate": rate,
                        "analysis_sample_rate": args.target_sample_rate,
                        "target_sample_rate": args.target_sample_rate,
                        "source_bandwidth_hz": rate // 2,
                        "task_mode": "vocoder_bwe",
                    }
                    manifest_rows.append(row)
                    feature_paths.append(feature_path)
                    variant_reports.append(
                        {
                            **row,
                            "status": status,
                            "source_length_error_samples": len(low_rate) - expected_source_samples,
                            "analysis_length_error_samples": len(analysis_waveform) - len(target),
                            "feature_target_error_samples": features.shape[0] * args.hop_length - len(target),
                            "quality": quality,
                        }
                    )
                    print(
                        f"[{file_index}/{len(sources)}] {status}: {relative} "
                        f"rate={rate} frames={frame_count}"
                    )
                except Exception as error:
                    variant_reports.append(
                        {"source_sample_rate": rate, "status": "failed", "error": str(error)}
                    )
                    print(f"[{file_index}/{len(sources)}] failed: {relative} rate={rate}: {error}")
                    if args.fail_fast:
                        raise
            target_quality = analyze_quality(target)
            target_quality["warnings"] = target_warnings + target_quality["warnings"]
            file_reports.append(
                {
                    "id": item_id,
                    "file": relative.as_posix(),
                    "native_sample_rate": native_rate,
                    "target_quality": target_quality,
                    "variants": variant_reports,
                }
            )
        except Exception as error:
            file_reports.append({"id": item_id, "file": relative.as_posix(), "error": str(error)})
            print(f"[{file_index}/{len(sources)}] failed master: {relative}: {error}")
            if args.fail_fast:
                raise

    if not feature_paths:
        raise SystemExit("preprocessing failed: no usable feature variants")
    validation_ids = _split_ids(sorted({row["id"] for row in manifest_rows}), args.validation_split, args.seed)
    train_rows = [row for row in manifest_rows if row["id"] not in validation_ids]
    valid_rows = [row for row in manifest_rows if row["id"] in validation_ids]
    _write_jsonl(output_root / "manifests/train.jsonl", train_rows)
    _write_jsonl(output_root / "manifests/valid.jsonl", valid_rows)

    stats = compute_feature_stats(feature_paths, sample_limit=args.sample_limit)
    np.savez(output_root / "feature_stats.npz", **stats)
    failed = sum(
        variant.get("status") == "failed"
        for report in file_reports for variant in report.get("variants", [])
    ) + sum("error" in report and "variants" not in report for report in file_reports)
    warnings_count = sum(
        len(variant.get("quality", {}).get("warnings", []))
        for report in file_reports for variant in report.get("variants", [])
    ) + sum(len(report.get("target_quality", {}).get("warnings", [])) for report in file_reports)
    report = {
        "config": {
            "task_mode": "vocoder_bwe",
            "source_sample_rates": list(args.source_sample_rates),
            "analysis_sample_rate": args.target_sample_rate,
            "target_sample_rate": args.target_sample_rate,
            "hop_length": args.hop_length,
            "f0_backend": args.f0_backend,
            "seed": args.seed,
            "validation_split": args.validation_split,
            "resampler": "scipy.signal.resample_poly/kaiser-beta-8.6",
            "roundtrip_delay_samples": {
                str(rate): measure_roundtrip_delay(rate, args.target_sample_rate)
                for rate in args.source_sample_rates
            },
        },
        "summary": {
            "masters": len(sources),
            "variants": len(manifest_rows),
            "train_variants": len(train_rows),
            "valid_variants": len(valid_rows),
            "failed": failed,
            "warnings": warnings_count,
            "frames": int(stats["count"]),
        },
        "files": file_reports,
    }
    report_path = output_root / "preprocess_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train manifest: {output_root / 'manifests/train.jsonl'} ({len(train_rows)})")
    print(f"valid manifest: {output_root / 'manifests/valid.jsonl'} ({len(valid_rows)})")
    print(f"statistics: {output_root / 'feature_stats.npz'}")
    print(f"report: {report_path}")
    if failed:
        raise SystemExit(f"completed with {failed} failed item(s)")


if __name__ == "__main__":
    main()
