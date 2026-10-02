from __future__ import annotations

import numpy as np


def analyze_quality(
    waveform: np.ndarray,
    features: np.ndarray | None = None,
    *,
    clipping_threshold: float = 0.999,
    silence_db: float = -55.0,
    f0_jump_semitones: float = 7.0,
) -> dict:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(values).all())
    safe = np.nan_to_num(values)
    rms = float(np.sqrt(np.mean(np.square(safe), dtype=np.float64))) if len(safe) else 0.0
    rms_db = float(20 * np.log10(max(rms, 1e-12)))
    report = {
        "finite": finite,
        "peak": float(np.max(np.abs(safe))) if len(safe) else 0.0,
        "rms_dbfs": rms_db,
        "clipping_ratio": float(np.mean(np.abs(safe) >= clipping_threshold)) if len(safe) else 0.0,
        "silent": rms_db < silence_db,
    }
    if features is not None:
        matrix = np.asarray(features)
        if matrix.ndim != 2 or 72 not in matrix.shape:
            raise ValueError("features must have shape [T,72] or [72,T]")
        if matrix.shape[0] == 72:
            matrix = matrix.T
        f0 = matrix[:, 1]
        voiced = (matrix[:, 0] > 0.5) & (f0 > 0) & np.isfinite(f0)
        pairs = voiced[1:] & voiced[:-1]
        jumps = np.zeros(max(0, len(f0) - 1), dtype=bool)
        jumps[pairs] = np.abs(12 * np.log2(f0[1:][pairs] / f0[:-1][pairs])) > f0_jump_semitones
        report.update(
            feature_finite=bool(np.isfinite(matrix).all()),
            voiced_ratio=float(voiced.mean()) if len(voiced) else 0.0,
            f0_jump_count=int(jumps.sum()),
            f0_jump_ratio=float(jumps.sum() / max(1, pairs.sum())),
        )
    warnings = []
    if not report["finite"]:
        warnings.append("non_finite_audio")
    if report["clipping_ratio"] > 1e-4:
        warnings.append("clipping")
    if report["silent"]:
        warnings.append("silence")
    if report.get("feature_finite") is False:
        warnings.append("non_finite_features")
    if report.get("f0_jump_ratio", 0.0) > 0.02:
        warnings.append("frequent_f0_jumps")
    report["warnings"] = warnings
    return report
