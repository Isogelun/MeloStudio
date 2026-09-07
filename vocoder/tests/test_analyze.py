import numpy as np

from vocoder.analyze import _fit_f0_length, _jobs, extract_f0


def test_single_file_job(tmp_path):
    source = tmp_path / "voice.wav"
    source.touch()
    assert list(_jobs(source, None)) == [(source, tmp_path / "voice.npy")]


def test_directory_jobs_mirror_tree(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "features"
    nested = source / "singer"
    nested.mkdir(parents=True)
    (nested / "b.wav").touch()
    (nested / "a.wav").touch()
    assert list(_jobs(source, output)) == [
        (nested / "a.wav", output / "singer" / "a.npy"),
        (nested / "b.wav", output / "singer" / "b.npy"),
    ]


def test_f0_length_is_exact():
    np.testing.assert_array_equal(
        _fit_f0_length(np.array([100.0, 101.0], dtype=np.float32), 4),
        np.array([100.0, 101.0, 0.0, 0.0], dtype=np.float32),
    )


def test_unknown_f0_backend_is_rejected():
    try:
        extract_f0(np.zeros(100, dtype=np.float32), 48_000, 256, 1, backend="bad")
    except ValueError as error:
        assert "unsupported F0 backend" in str(error)
    else:
        raise AssertionError("unknown backend should fail")
