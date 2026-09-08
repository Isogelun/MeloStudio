import numpy as np

from vocoder.compare_f0 import compare_contours


def test_compare_contours_reports_voicing_and_pitch_error():
    first = np.array([0, 100, 200, 400], dtype=np.float32)
    second = np.array([0, 100, 0, 200], dtype=np.float32)
    result = compare_contours(first, second)
    assert result["frames"] == 4
    assert result["voicing_agreement"] == 0.75
    assert result["overlap_voiced_frames"] == 2
    assert result["median_absolute_cents"] == 600.0
    assert result["gross_pitch_error_ratio"] == 0.5
