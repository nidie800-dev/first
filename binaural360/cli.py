"""Command line entry point for the 360 binaural ASMR renderer.

Examples
--------
Render a wav file, source fixed 30 deg to the right, intimate distance:
    python -m binaural360.cli in.wav out.wav --az 30 --dist 0.4

Full 360 orbit, one lap every 8 s, small booth room:
    python -m binaural360.cli in.wav out.wav --rotate 8 --room booth

Use measured HRIRs (SOFA / .mat / .npz):
    python -m binaural360.cli in.wav out.wav --hrtf subject.sofa --az 45

No real dataset? use the built-in synthetic grid to test the HRTF path:
    python -m binaural360.cli in.wav out.wav --hrtf-synthetic --rotate 6

No input file? generate a synthetic test and render it:
    python -m binaural360.cli --demo out.wav --rotate 6
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.io import wavfile

from .hrtf import load_mat, load_sofa, synthetic_hrtf
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


def _load_hrtf_arg(path: str, flip: str, interp: str):
    ext = os.path.splitext(path)[1].lower()
    flip_flag = None if flip == "auto" else (flip == "yes")
    if ext == ".sofa":
        kw = {} if flip_flag is None else {"azimuth_flip": flip_flag}
        data = load_sofa(path, **kw)
    elif ext == ".mat":
        kw = {} if flip_flag is None else {"azimuth_flip": flip_flag}
        data = load_mat(path, **kw)
    else:  # .npz
        from .hrtf import load_hrtf
        data = load_hrtf(path)
    if interp != data.interp:
        data.interp = interp
        if interp == "nearest":
            data.k = 1
    return data


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

    g = ap.add_argument_group("measured HRTF")
    g.add_argument("--hrtf", metavar="PATH", help=".sofa / .mat / .npz HRIR set")
    g.add_argument("--hrtf-synthetic", action="store_true",
                   help="use the built-in synthetic HRIR grid (for testing)")
    g.add_argument("--hrtf-interp", default="idw", choices=["idw", "nearest"],
                   help="direction interpolation across measured HRIRs")
    g.add_argument("--hrtf-flip", default="auto", choices=["auto", "yes", "no"],
                   help="flip azimuth sign (SOFA default yes, .mat default no)")

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

    if args.hrtf_synthetic:
        cfg.hrtf = synthetic_hrtf(sr=cfg.sr)
    elif args.hrtf:
        cfg.hrtf = _load_hrtf_arg(args.hrtf, args.hrtf_flip, args.hrtf_interp)

    if args.demo:
        mono = make_test_signal(6.0, sr=cfg.sr)
        out = render_binaural(mono, cfg, sr=cfg.sr)
        _write_wav(args.demo, cfg.sr, out)
        model = "measured HRTF" if cfg.hrtf is not None else "parametric"
        print(f"[demo] wrote {args.demo}  ({out.shape[0]} samples, {cfg.sr} Hz, "
              f"stereo, {model})")
        return

    if not args.input or not args.output:
        ap.error("provide INPUT OUTPUT, or use --demo OUT")

    sr_in, data = _read_wav(args.input)
    out = render_binaural(data, cfg, sr=sr_in)
    _write_wav(args.output, cfg.sr, out)
    print(f"[ok] {args.input} -> {args.output}  ({out.shape[0]} samples, {cfg.sr} Hz)")


if __name__ == "__main__":
    main()
