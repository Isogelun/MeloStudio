import json
import sys

import numpy as np
import pytest

from melo.vocoder.preprocessing import preprocess


@pytest.mark.parametrize("fail_fast", [False, True])
def test_failed_file_preserves_error_and_respects_fail_fast(tmp_path, monkeypatch, capsys, fail_fast):
    source = tmp_path / "input"
    source.mkdir()
    (source / "a_bad.wav").touch()
    (source / "b_good.wav").touch()
    output = tmp_path / "output"
    visited = []
    original_error = RuntimeError("cannot decode audio")

    def read_audio(path):
        visited.append(path.name)
        if path.name == "a_bad.wav":
            raise original_error
        return np.zeros(2048, dtype=np.float32), 48_000

    features = np.zeros((8, 72), dtype=np.float32)
    features[:, 1] = 120.0
    features[:, 2] = 1.0
    monkeypatch.setattr(preprocess, "read_audio", read_audio)
    monkeypatch.setattr(preprocess, "F0Extractor", lambda *args: object())
    monkeypatch.setattr(preprocess, "analyze_waveform", lambda *args, **kwargs: features)
    argv = ["nhn-preprocess", str(source), str(output)]
    if fail_fast:
        argv.append("--fail-fast")
    monkeypatch.setattr(sys, "argv", argv)

    if fail_fast:
        with pytest.raises(RuntimeError) as caught:
            preprocess.main()
        assert caught.value is original_error
        assert visited == ["a_bad.wav"]
        assert not (output / "b_good.npy").exists()
    else:
        with pytest.raises(SystemExit, match="completed with 1 failed file"):
            preprocess.main()
        assert visited == ["a_bad.wav", "b_good.wav"]
        np.testing.assert_array_equal(np.load(output / "b_good.npy"), features)
        assert (output / "feature_stats.npz").is_file()
        report = json.loads((output / "preprocess_report.json").read_text())
        assert report["summary"]["failed"] == 1
        assert report["summary"]["processed"] == 1
        assert report["files"][0]["error"] == str(original_error)

    assert "[1/2] failed: a_bad.wav: cannot decode audio" in capsys.readouterr().out
