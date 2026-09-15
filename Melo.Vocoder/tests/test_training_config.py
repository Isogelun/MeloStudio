from pathlib import Path

import pytest

from vocoder.training.post_train import build_parser as build_post_parser
from vocoder.training.train import build_parser as build_train_parser
from vocoder.training.config import (
    detect_training_section,
    parse_training_args,
    require_configured_paths,
    resolve_training_device,
)


def test_separate_yaml_files_supply_training_stages_and_cli_overrides(tmp_path):
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        """
config_type: nhn_base_train
description: Train the full base vocoder.
common:
  device: cpu
  seed: 7
base_train:
  data: data/base
  output: runs/base
  batch_size: 4
""".strip(),
        encoding="utf-8",
    )
    base = parse_training_args(
        build_train_parser(),
        "base_train",
        ["--config", str(base_config), "--batch-size", "2"],
    )
    assert base.data == tmp_path / "data/base"
    assert base.output == tmp_path / "runs/base"
    assert base.batch_size == 2
    assert base.device == "cpu"
    assert base.seed == 7

    post_config = tmp_path / "post.yaml"
    post_config.write_text(
        """
config_type: nhn_dsp_post_train
description: Freeze base and train the DSP head.
common:
  device: cpu
post_train:
  data: data/base
  base_checkpoint: runs/base/best.pt
  output: runs/post
  epochs: 3
""".strip(),
        encoding="utf-8",
    )
    post = parse_training_args(
        build_post_parser(), "post_train", ["--config", str(post_config)]
    )
    assert post.base_checkpoint == tmp_path / "runs/base/best.pt"
    assert post.output == tmp_path / "runs/post"
    assert post.epochs == 3


def test_yaml_rejects_unknown_key(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "config_type: nhn_base_train\n"
        "description: Invalid option test.\n"
        "base_train:\n  batch_szie: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        parse_training_args(
            build_train_parser(), "base_train", ["--config", str(config)]
        )


def test_yaml_rejects_wrong_training_stage(tmp_path):
    config = tmp_path / "post.yaml"
    config.write_text(
        "config_type: nhn_dsp_post_train\n"
        "description: DSP only.\n"
        "post_train:\n  epochs: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        parse_training_args(
            build_train_parser(), "base_train", ["--config", str(config)]
        )


def test_required_paths_can_still_come_from_positionals():
    parser = build_train_parser()
    args = parse_training_args(parser, "base_train", ["data", "output"])
    require_configured_paths(parser, args, ("data", "output"))
    assert args.data == Path("data")


def test_explicit_training_device_is_preserved():
    assert resolve_training_device("cpu") == "cpu"


def test_config_type_selects_internal_trainer(tmp_path):
    config = tmp_path / "post.yaml"
    config.write_text(
        "config_type: nhn_dsp_post_train\n"
        "description: Select DSP trainer.\n"
        "post_train: {}\n",
        encoding="utf-8",
    )
    assert detect_training_section(["--config", str(config)]) == "post_train"
    assert detect_training_section([]) == "base_train"
