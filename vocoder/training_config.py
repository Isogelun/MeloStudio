from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml
import torch


CONFIG_TYPES = {
    "base_train": "nhn_base_train",
    "post_train": "nhn_dsp_post_train",
}


def detect_training_section(argv: Sequence[str] | None = None) -> str:
    """Choose the internal trainer from a typed YAML file; legacy CLI means base."""

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    known, _ = probe.parse_known_args(argv)
    if known.config is None:
        return "base_train"
    config_path = known.config.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        probe.error(f"{config_path} must contain a YAML mapping")
    actual_type = payload.get("config_type")
    inverse = {value: key for key, value in CONFIG_TYPES.items()}
    if actual_type not in inverse:
        probe.error(
            f"unsupported config_type={actual_type!r}; expected one of "
            f"{', '.join(sorted(inverse))}"
        )
    return inverse[actual_type]


def parse_training_args(
    parser: argparse.ArgumentParser,
    section: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Load YAML defaults, then let explicit command-line values override them."""

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    known, _ = probe.parse_known_args(argv)
    if known.config is None:
        return parser.parse_args(argv)

    config_path = known.config.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        parser.error(f"{config_path} must contain a YAML mapping")
    expected_type = CONFIG_TYPES[section]
    actual_type = payload.get("config_type")
    if actual_type != expected_type:
        parser.error(
            f"{config_path} has config_type={actual_type!r}; "
            f"{section} requires {expected_type!r}"
        )
    description = payload.get("description")
    if not isinstance(description, str) or not description.strip():
        parser.error(f"{config_path} must declare a non-empty description")
    common = payload.get("common", {}) or {}
    selected = payload.get(section, {}) or {}
    if not isinstance(common, dict) or not isinstance(selected, dict):
        parser.error(f"common and {section} must be YAML mappings")
    values = {**common, **selected}
    values = {str(key).replace("-", "_"): value for key, value in values.items()}

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(values) - set(actions))
    if unknown:
        parser.error(f"unknown keys in [{section}]: {', '.join(unknown)}")

    defaults = {}
    for key, value in values.items():
        if value is None:
            continue
        action = actions[key]
        if action.type is Path:
            path = Path(value)
            defaults[key] = path if path.is_absolute() else (config_path.parent / path).resolve()
        elif action.type is not None and not isinstance(value, action.type):
            defaults[key] = action.type(value)
        else:
            defaults[key] = value
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def require_configured_paths(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    names: tuple[str, ...],
) -> None:
    missing = [name for name in names if getattr(args, name, None) is None]
    if missing:
        flags = ", ".join(missing)
        parser.error(f"missing required training paths: {flags}; pass them or set them in YAML")


def resolve_training_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
