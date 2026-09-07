import numpy as np
import torch

from vocoder.audio import write_wav
from vocoder.data import LLSMWavDataset
from vocoder.stats import compute_feature_stats, load_feature_range


def test_dataset_center_crop_and_alignment(tmp_path):
    features = np.zeros((10, 72), dtype=np.float32)
    features[:, 1] = np.arange(10, dtype=np.float32)
    np.save(tmp_path / "item.npy", features)
    waveform = np.arange(40, dtype=np.float32) / 100.0
    write_wav(tmp_path / "item.wav", waveform, 8_000)
    dataset = LLSMWavDataset(
        tmp_path, sample_rate=8_000, hop_length=4, segment_frames=4, random_crop=False
    )
    cropped_features, cropped_waveform = dataset[0]
    torch.testing.assert_close(cropped_features[:, 1], torch.arange(3, 7).float())
    assert cropped_waveform.shape == (1, 16)
    torch.testing.assert_close(
        cropped_waveform.squeeze(),
        torch.from_numpy((waveform[12:28] * 32767).astype("<i2").astype(np.float32) / 32768.0),
    )


def test_feature_stats_round_trip(tmp_path):
    first = np.tile(np.arange(72, dtype=np.float32), (5, 1))
    second = first + 2.0
    first_path, second_path = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    stats = compute_feature_stats([first_path, second_path])
    assert int(stats["count"]) == 10
    np.testing.assert_allclose(stats["mean"], np.arange(72) + 1.0)
    output = tmp_path / "stats.npz"
    np.savez(output, **stats)
    minimum, maximum = load_feature_range(output)
    assert minimum.shape == maximum.shape == (72,)
    assert np.all(maximum > minimum)
