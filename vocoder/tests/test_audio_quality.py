import json

import numpy as np
import soundfile as sf

from vocoder.preprocessing.audio import read_audio, write_wav
from vocoder.preprocessing.preprocess_bwe import parse_sample_rates
from vocoder.preprocessing.quality import analyze_quality
from vocoder.preprocessing.resample import bandwidth_degrade, measure_roundtrip_delay, resample_audio
from vocoder.training.data import LLSMWavDataset


def test_audio_reads_24_bit_wav_and_flac(tmp_path):
    waveform = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    wav, flac = tmp_path / "audio.wav", tmp_path / "audio.flac"
    sf.write(wav, waveform, 24_000, subtype="PCM_24")
    sf.write(flac, waveform, 24_000)
    for path in (wav, flac):
        decoded, sample_rate = read_audio(path)
        assert sample_rate == 24_000
        np.testing.assert_allclose(decoded, waveform, atol=2e-4)


def test_resampling_and_bandwidth_degrade_have_exact_lengths():
    waveform = np.random.default_rng(7).normal(size=48_001).astype(np.float32)
    low, restored = bandwidth_degrade(waveform, 16_000)
    assert low.shape == (16_000,)
    assert restored.shape == waveform.shape
    assert resample_audio(low, 16_000, 48_000, target_length=123).shape == (123,)
    assert abs(measure_roundtrip_delay(16_000)) <= 1


def test_quality_report_flags_clipping_silence_and_f0_jumps():
    features = np.zeros((8, 72), dtype=np.float32)
    features[:, 0] = 1
    features[:, 1] = [100, 100, 100, 400, 400, 400, 400, 400]
    report = analyze_quality(np.ones(1000, dtype=np.float32), features)
    assert "clipping" in report["warnings"]
    assert report["f0_jump_count"] == 1
    assert "frequent_f0_jumps" in report["warnings"]
    assert analyze_quality(np.zeros(1000, dtype=np.float32))["silent"]


def test_manifest_dataset_groups_rate_variants(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "targets").mkdir()
    features = np.zeros((4, 72), dtype=np.float32)
    np.save(tmp_path / "features/item@16000.npy", features)
    np.save(tmp_path / "features/item@48000.npy", features)
    write_wav(tmp_path / "targets/item.wav", np.zeros(1024, dtype=np.float32), 48_000)
    rows = [
        {"id": "item", "features": f"features/item@{rate}.npy", "target": "targets/item.wav"}
        for rate in (16_000, 48_000)
    ]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    grouped = LLSMWavDataset(tmp_path, manifest=manifest, group_variants=True)
    ungrouped = LLSMWavDataset(tmp_path, manifest=manifest, group_variants=False)
    assert len(grouped) == 1
    assert len(ungrouped) == 2
    assert grouped[0][0].shape == (4, 72)


def test_parse_sample_rates_deduplicates_in_order():
    assert parse_sample_rates("48000,16000,16000") == (48_000, 16_000)
