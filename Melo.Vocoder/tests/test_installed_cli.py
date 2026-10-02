"""Exercise installed entry points outside the source checkout."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from importlib.metadata import distribution

import pytest


SCRIPTS = [
    entry.name
    for entry in distribution("melostudio-nhn-vocoder").entry_points
    if entry.group == "console_scripts"
]


def run_help(command, cwd):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [*command, "--help"], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("name", SCRIPTS)
def test_installed_console_script(name, tmp_path):
    suffix = ".exe" if os.name == "nt" else ""
    script = Path(sysconfig.get_path("scripts")) / (name + suffix)
    run_help([str(script)], tmp_path)


@pytest.mark.parametrize("command", [[], ["preprocess"], ["train"], ["export"], ["infer"]])
def test_module_cli(command, tmp_path):
    run_help([sys.executable, "-m", "melo.vocoder", *command], tmp_path)
