"""Emotion control layer for TTS (backend agnostic).

This module is the "emotion TTS" half of the project, decoupled from any
particular model:

  1. parse a lightweight inline markup into ordered segments
       - emotions:    [开心] [生气] [悲伤] [惊讶] [平静]   (also English)
       - intensity:   [开心:0.8]  or  [开心*0.8]
       - nonverbal:   [笑] [叹] [哭] [耳语]
       - mid-sentence switching works because tags can appear anywhere
  2. adapt the segments to a concrete backend:
       - to_fish_tags()      -> inline cue text for OpenAudio / Fish style models
       - to_instructions()   -> per-segment instruction for CosyVoice2 style models
  3. synthesise by calling an injected synth(Segment) -> np.ndarray, then
     optionally render to 360 binaural via binaural360.renderer.

No model weights are bundled. `make_mock_synth()` is a placeholder that turns
each segment into a tone, only to exercise the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Segment",
    "EmotionEngine",
    "parse_emotions",
    "EMO_INFO",
    "make_mock_synth",
]

# canonical emotion -> display / backend metadata
EMO_INFO = {
    "neutral":   dict(cn="平静", fish=None,          pitch=200.0),
    "happy":     dict(cn="开心", fish="happy",       pitch=320.0),
    "angry":     dict(cn="生气", fish="angry",       pitch=250.0),
    "sad":       dict(cn="悲伤", fish="sad",         pitch=175.0),
    "surprised": dict(cn="惊讶", fish="surprised",   pitch=400.0),
    "calm":      dict(cn="平静", fish="calm",        pitch=195.0),
}

_ALIASES = {
    # 喜 / 乐 -> happy
    "开心": "happy", "高兴": "happy", "喜": "happy", "乐": "happy", "快乐": "happy",
    "欢快": "happy", "happy": "happy", "joy": "happy",
    # 怒
    "生气": "angry", "愤怒": "angry", "怒": "angry", "angry": "angry", "anger": "angry",
    # 哀
    "悲伤": "sad", "难过": "sad", "哀": "sad", "sad": "sad", "sorrow": "sad",
    # 惊
    "惊讶": "surprised", "惊": "surprised", "震惊": "surprised",
    "surprise": "surprised", "surprised": "surprised",
    # calm / neutral
    "平静": "calm", "冷静": "calm", "温柔": "calm", "calm": "calm", "soft": "calm",
    "中性": "neutral", "平常": "neutral", "neutral": "neutral", "normal": "neutral",
}

_NV_ALIASES = {
    "笑": "laugh", "笑声": "laugh", "哈哈": "laugh", "laugh": "laugh",
    "叹": "sigh", "叹气": "sigh", "sigh": "sigh",
    "哭": "cry", "哭泣": "cry", "cry": "cry",
    "耳语": "whisper", "低语": "whisper", "气声": "whisper", "悄悄话": "whisper",
    "asmr": "whisper", "whisper": "whisper",
}

_NV_FISH = {"laugh": "laugh", "sigh": "sigh", "cry": "cry", "whisper": "whisper"}
_NV_INSTR = {
    "laugh": "，带笑声", "sigh": "，带叹气",
    "cry": "，带哭腔", "whisper": "，用气声耳语",
}

_TAG_RE = re.compile(r"\[([^\]]*)\]")


@dataclass
class Segment:
    text: str
    emotion: str = "neutral"
    intensity: float = 0.6
    nonverbal: str | None = None
    meta: dict = field(default_factory=dict)


class EmotionEngine:
    def __init__(self, default_emotion: str = "neutral", default_intensity: float = 0.6):
        if default_emotion not in EMO_INFO:
            raise ValueError(f"unknown default emotion: {default_emotion}")
        self.default_emotion = default_emotion
        self.default_intensity = float(np.clip(default_intensity, 0.0, 1.0))

    # ---- parsing ------------------------------------------------------
    @staticmethod
    def _parse_tag(inner: str):
        """Return (canonical, kind, value) where kind in {emotion, nonverbal}."""
        inner = inner.strip()
        value = None
        for sep in (":", "*", "："):
            if sep in inner:
                inner, value = inner.split(sep, 1)
                inner = inner.strip()
                break
        key = inner.strip().lower()
        try:
            value = float(value) if value is not None else None
        except ValueError:
            value = None
        if key in _ALIASES:
            return _ALIASES[key], "emotion", value
        if key in _NV_ALIASES:
            return _NV_ALIASES[key], "nonverbal", value
        return None, None, None

    def parse(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        emotion = self.default_emotion
        intensity = self.default_intensity
        nonverbal = None
        pos = 0

        for m in _TAG_RE.finditer(text):
            chunk = text[pos:m.start()]
            if chunk.strip():
                segments.append(Segment(chunk.strip(), emotion,
                                        float(np.clip(intensity, 0.0, 1.0)), nonverbal))
                nonverbal = None          # a nonverbal cue applies once
            canonical, kind, value = self._parse_tag(m.group(1))
            if kind == "emotion":
                emotion = canonical
                if value is not None:
                    intensity = value
            elif kind == "nonverbal":
                nonverbal = canonical
                if value is not None:
                    intensity = value
            pos = m.end()

        tail = text[pos:]
        if tail.strip():
            segments.append(Segment(tail.strip(), emotion,
                                    float(np.clip(intensity, 0.0, 1.0)), nonverbal))
        if not segments:                 # empty / tag-only input
            segments.append(Segment("", emotion, self.default_intensity, None))
        return segments

    @staticmethod
    def _as_segments(src) -> list[Segment]:
        if isinstance(src, str):
            return EmotionEngine().parse(src)
        return list(src)

    # ---- backend adapters --------------------------------------------
    def to_fish_tags(self, src) -> str:
        """Inline cue string for OpenAudio / Fish style models."""
        parts = []
        for s in self._as_segments(src):
            cue = EMO_INFO[s.emotion]["fish"]
            if cue:
                parts.append(f"[{cue}]")
            if s.nonverbal:
                parts.append(f"[{_NV_FISH[s.nonverbal]}]")
            if s.text:
                parts.append(s.text)
        return " ".join(parts)

    def to_instructions(self, src) -> list[dict]:
        """Per-segment instruction for CosyVoice2 style instruction models."""
        out = []
        for s in self._as_segments(src):
            if s.intensity <= 0.33:
                adv = "略带"
            elif s.intensity <= 0.66:
                adv = "用"
            else:
                adv = "非常"
            instruction = f"{adv}{EMO_INFO[s.emotion]['cn']}的语气" + _NV_INSTR.get(s.nonverbal, "")
            out.append({
                "text": s.text,
                "emotion": s.emotion,
                "intensity": s.intensity,
                "nonverbal": s.nonverbal,
                "instruction": instruction,
            })
        return out

    # ---- synthesis glue ----------------------------------------------
    def synthesize(self, src, synth, sr: int = 48000, gap_ms: float = 120.0) -> np.ndarray:
        """Synthesise each segment with synth(Segment) and concatenate."""
        segments = self.parse(src) if isinstance(src, str) else list(src)
        gap = np.zeros(max(0, int(sr * gap_ms / 1000.0)))
        chunks: list[np.ndarray] = []
        for s in segments:
            if not s.text and s.nonverbal is None:
                continue
            wav = np.asarray(synth(s), dtype=np.float64).reshape(-1)
            if wav.size:
                chunks.append(wav)
                chunks.append(gap)
        if not chunks:
            return np.zeros(0)
        return np.concatenate(chunks[:-1])       # drop trailing silence

    def synthesize_spatial(self, src, synth, sr: int = 48000, binaural_cfg=None,
                           gap_ms: float = 120.0) -> np.ndarray:
        """Synthesise then render to 360 binaural stereo."""
        from .renderer import BinauralConfig, render_binaural

        mono = self.synthesize(src, synth, sr=sr, gap_ms=gap_ms)
        cfg = binaural_cfg or BinauralConfig(sr=sr)
        return render_binaural(mono, cfg, sr=sr)


def parse_emotions(text: str) -> list[Segment]:
    return EmotionEngine().parse(text)


def make_mock_synth(sr: int = 48000, seed: int = 0):
    """Placeholder synth: turns a Segment into a tone whose pitch follows the
    emotion. NOT a real TTS -- only for testing the pipeline end to end."""
    rng = np.random.default_rng(seed)

    def synth(seg: Segment) -> np.ndarray:
        dur = 0.10 * max(len(seg.text), 1) + 0.15
        t = np.arange(max(1, int(dur * sr))) / sr
        f0 = EMO_INFO.get(seg.emotion, EMO_INFO["neutral"])["pitch"] * (0.8 + 0.4 * seg.intensity)
        if seg.nonverbal == "whisper":
            x = rng.standard_normal(t.size) * 0.5
        else:
            x = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
            x *= 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t)
            if seg.emotion == "angry":
                x += 0.2 * np.sin(2 * np.pi * 0.5 * f0 * t) * np.sin(2 * np.pi * 30.0 * t)
        env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (dur - t) / 0.03)
        return (x * env * (0.3 + 0.7 * seg.intensity)).astype(np.float64)

    return synth


def _main() -> None:  # small demo CLI
    import argparse

    ap = argparse.ArgumentParser(description="emotion layer demo")
    ap.add_argument("--text", default="[开心:0.9]今天见到你真好呀。[笑]你呢？"
                                     "[耳语:0.8]不过……我有点[悲伤:0.7]想哭。")
    ap.add_argument("--out", help="mock-synthesise + binaural render to this wav")
    ap.add_argument("--rotate", type=float, default=8.0)
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()

    eng = EmotionEngine()
    segs = eng.parse(args.text)
    print("== segments ==")
    for i, s in enumerate(segs):
        print(f"  {i}: emo={s.emotion:9s} int={s.intensity:.2f} "
              f"nv={str(s.nonverbal):7s} text={s.text!r}")
    print("\n== fish inline tags ==")
    print("  " + eng.to_fish_tags(segs))
    print("\n== cosyvoice-style instructions ==")
    for d in eng.to_instructions(segs):
        print(f"  {d['instruction']}  <- {d['text']!r}")

    if args.out:
        from scipy.io import wavfile
        from .renderer import BinauralConfig

        cfg = BinauralConfig(sr=args.sr, rotate_period_s=args.rotate, distance_m=0.4)
        stereo = eng.synthesize_spatial(segs, make_mock_synth(args.sr), sr=args.sr,
                                        binaural_cfg=cfg)
        wavfile.write(args.out, args.sr,
                      (np.clip(stereo, -1, 1) * 32767).astype(np.int16))
        print(f"\n[demo] wrote {args.out}  ({stereo.shape[0]} samples, stereo)")


if __name__ == "__main__":
    _main()
