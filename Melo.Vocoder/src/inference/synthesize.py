from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..preprocessing.audio import AUDIO_SUFFIXES
from ..sdk import VocoderSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize fixed-48-kHz WAV from LLSM features or <=48-kHz audio"
    )
    parser.add_argument("input", type=Path, help="WAV/FLAC/NPY/NPZ input")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--f0-backend", choices=("fcpe", "rmvpe", "parselmouth"), default="fcpe")
    parser.add_argument("--f0-device", default="cpu")
    parser.add_argument("--chunk-frames", type=int)
    parser.add_argument("--overlap-frames", type=int, default=32)
    parser.add_argument("--strict-bandwidth", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = VocoderSession.from_checkpoint(
        args.checkpoint,
        device=args.device,
        seed=args.seed,
        strict_bandwidth=args.strict_bandwidth,
    )
    if args.input.suffix.lower() in AUDIO_SUFFIXES:
        result = session.synthesize_waveform(
            args.input,
            f0_backend=args.f0_backend,
            f0_device=args.f0_device,
            chunk_frames=args.chunk_frames,
            overlap_frames=args.overlap_frames,
            seed=args.seed,
        )
    else:
        result = (
            session.synthesize_chunked(
                args.input,
                chunk_frames=args.chunk_frames,
                overlap_frames=args.overlap_frames,
                seed=args.seed,
            )
            if args.chunk_frames is not None
            else session.synthesize(args.input, seed=args.seed)
        )
    summary = {
        "sample_rate": result.sample_rate,
        "samples": len(result.samples),
        "seconds": result.duration_seconds,
        **result.metadata,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if not args.validate_only:
        result.save(args.output)
        print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
