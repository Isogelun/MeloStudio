"""NHN vocoder: a compact harmonic-plus-noise neural waveform generator."""

from .config import NHNVocoderConfig
from .model import NHNVocoder

__all__ = ["NHNVocoder", "NHNVocoderConfig"]

