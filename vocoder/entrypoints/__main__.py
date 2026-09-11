"""Four-command entry point for the complete vocoder workflow."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


COMMANDS = {
    "preprocess": ("vocoder.entrypoints.preprocess", "Prepare training data"),
    "train": ("vocoder.entrypoints.train", "Run base or DSP post-training"),
    "export": ("vocoder.entrypoints.export", "Export an inference model"),
    "infer": ("vocoder.entrypoints.infer", "Run high-level inference"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nhn-vocoder",
        description="Unified NHN vocoder preprocessing, training, and inference CLI.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, add_help=False, help=description)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, forwarded = parser.parse_known_args(raw)
    if args.command is None:
        parser.print_help()
        return

    module_name, _ = COMMANDS[args.command]
    module = __import__(module_name, fromlist=["main"])
    command_main = module.main
    previous = sys.argv
    sys.argv = [f"nhn-vocoder {args.command}", *forwarded]
    try:
        command_main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
