"""Unified preprocessing entry for standard and bandwidth-extension datasets."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..preprocessing import preprocess, preprocess_bwe


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(
        prog="nhn-vocoder preprocess",
        add_help=False,
        description="Prepare standard LLSM72 data or paired 16→48 kHz BWE data.",
        epilog=(
            "Use '--pipeline standard --help' or '--pipeline bwe --help' "
            "to see pipeline-specific arguments."
        ),
    )
    selector.add_argument(
        "--pipeline",
        choices=("standard", "bwe"),
        default="standard",
        help="standard: WAV→LLSM72; bwe: degraded input→48 kHz target pairs",
    )
    selector.add_argument("-h", "--help", action="store_true")
    selected, forwarded = selector.parse_known_args(raw)
    pipeline_explicit = "--pipeline" in raw
    if selected.help and not pipeline_explicit:
        selector.print_help()
        return
    if selected.help:
        forwarded.append("--help")
    target = preprocess_bwe.main if selected.pipeline == "bwe" else preprocess.main

    previous = sys.argv
    sys.argv = [f"nhn-vocoder preprocess --pipeline {selected.pipeline}", *forwarded]
    try:
        target()
    finally:
        sys.argv = previous
