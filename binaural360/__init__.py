"""360 binaural ASMR renderer (independent DSP implementation)."""

from .emotion import EMO_INFO, EmotionEngine, Segment, make_mock_synth, parse_emotions
from .hrtf import HRTFData, load_hrtf, load_mat, load_sofa, synthetic_hrtf
from .renderer import BinauralConfig, make_test_signal, render_binaural

__all__ = [
    "BinauralConfig",
    "render_binaural",
    "make_test_signal",
    "HRTFData",
    "load_hrtf",
    "load_sofa",
    "load_mat",
    "synthetic_hrtf",
    "Segment",
    "EmotionEngine",
    "parse_emotions",
    "make_mock_synth",
    "EMO_INFO",
]
