from __future__ import annotations

import argparse
from pathlib import Path

from .sdk import VocoderSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize WAV from 72-D LLSM .npy features")
    parser.add_argument("features", type=Path, help="float .npy array shaped [T,72] or [72,T]")
    parser.add_argument("checkpoint", type=Path, help="trained NHN .pt checkpoint")
    parser.add_argument("output", type=Path, help="output mono 16-bit WAV")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=1234, help="noise seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = VocoderSession.from_checkpoint(
        args.checkpoint, device=args.device, seed=args.seed
    )
    session.synthesize(args.features, seed=args.seed).save(args.output)


if __name__ == "__main__":
    main()
