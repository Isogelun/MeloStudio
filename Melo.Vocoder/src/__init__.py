"""NHN vocoder: a compact harmonic-plus-noise neural waveform generator."""

import importlib
import sys

from .core.config import NHNVocoderConfig
from .core.model import NHNVocoder
from .sdk import AudioResult, DSPControl, LLSMFeatures, SynthesisRequest, VocoderSession

# Compatibility for the original flat module layout. New code should use the
# responsibility-based subpackages below.
_COMPAT_MODULES = {
    "analyze": ".preprocessing.analyze",
    "audio": ".preprocessing.audio",
    "benchmark": ".inference.benchmark",
    "checkpoint": ".training.checkpoint",
    "compare_f0": ".preprocessing.compare_f0",
    "config": ".core.config",
    "data": ".training.data",
    "discriminators": ".training.discriminators",
    "dsp": ".core.dsp",
    "export": ".inference.export",
    "features": ".core.features",
    "infer": ".inference.infer",
    "layers": ".core.layers",
    "losses": ".training.losses",
    "model": ".core.model",
    "post_train": ".training.post_train",
    "preprocess": ".preprocessing.preprocess",
    "preprocess_bwe": ".preprocessing.preprocess_bwe",
    "quality": ".preprocessing.quality",
    "resample": ".preprocessing.resample",
    "stats": ".preprocessing.stats",
    "synthesize": ".inference.synthesize",
    "train": ".training.train",
    "training_config": ".training.config",
}
for _old_name, _new_name in _COMPAT_MODULES.items():
    sys.modules[f"{__name__}.{_old_name}"] = importlib.import_module(_new_name, __name__)

__all__ = [
    "AudioResult", "DSPControl", "LLSMFeatures", "NHNVocoder", "NHNVocoderConfig",
    "SynthesisRequest", "VocoderSession"
]
