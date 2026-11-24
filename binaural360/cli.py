"""Command line entry point for the 360 binaural ASMR renderer.

Examples
--------
Render a wav file, source fixed 30 deg to the right, intimate distance:
    python -m binaural360.cli in.wav out.wav --az 30 --dist 0.4

Full 360 orbit, one lap every 8 s, small booth room:
    python -m binaural360.cli in.wav out.wav --rotate 8 --room booth

No input file? generate a synthetic test and render it:
    python -m binaural360.cli --demo out.wav --rotate 6
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.io import wavfile

from .renderer import BinauralConfig, make_test_signal, render_binaural


def _read_wav(path: str) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)
    return int(sr), data


def _write_wav(path: str, sr: int, stereo: np.ndarray) -> None:
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    wavfile.write(path, sr, pcm)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="360 binaural ASMR renderer")
    ap.add_argument("input", nargs="?", help="input mono/stereo wav")
    ap.add_argument("output", nargs="?", help="output binaural wav")
    ap.add_argument("--demo", metavar="OUT", help="render a synthetic test signal instead")
    ap.add_argument("--az", type=float, default=0.0, help="start azimuth deg (+=right)")
    ap.add_argument("--el", type=float, default=0.0, help="elevation deg")
    ap.add_argument("--dist", type=float, default=0.55, help="distance in metres")
    ap.add_argument("--rotate", type=float, default=None,
                    help="seconds per full 360 orbit (omit for static)")
    ap.add_argument("--direction", type=int, default=1, choices=[1, -1],
                    help="orbit direction")
    ap.add_argument("--room", default="booth", choices=["dry", "booth", "room", "hall"])
    ap.add_argument("--wet", type=float, default=0.18, help="reverb wet amount 0..1")
    ap.add_argument("--sr", type=int, default=48000, help="output sample rate")
    args = ap.parse_args(argv)

    cfg = BinauralConfig(
        sr=args.sr,
        start_azimuth_deg=args.az,
        elevation_deg=args.el,
        distance_m=args.dist,
        rotate_period_s=args.rotate,
        orbit_direction=args.direction,
        room=args.room,
        wet=args.wet,
    )

    if args.demo:
        mono = make_test_signal(6.0, sr=cfg.sr)
        out = render_binaural(mono, cfg, sr=cfg.sr)
        _write_wav(args.demo, cfg.sr, out)
        print(f"[demo] wrote {args.demo}  ({out.shape[0]} samples, {cfg.sr} Hz, stereo)")
        return

    if not args.input or not args.output:
        ap.error("provide INPUT OUTPUT, or use --demo OUT")

    sr_in, data = _read_wav(args.input)
    out = render_binaural(data, cfg, sr=sr_in)
    _write_wav(args.output, cfg.sr, out)
    print(f"[ok] {args.input} -> {args.output}  ({out.shape[0]} samples, {cfg.sr} Hz)")


if __name__ == "__main__":
    main()
