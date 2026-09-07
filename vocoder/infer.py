from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .audio import write_wav
from .checkpoint import load_checkpoint


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
    model = load_checkpoint(args.checkpoint, args.device)
    values = np.load(args.features, allow_pickle=False).astype(np.float32)
    features = torch.from_numpy(values).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    with torch.inference_mode():
        waveform = model(features, generator=generator).squeeze().cpu().numpy()
    write_wav(args.output, waveform, model.config.sample_rate)


if __name__ == "__main__":
    main()

