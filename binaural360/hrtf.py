"""Measured-HRTF support for the binaural renderer.

Load real HRIR sets and query them at arbitrary directions, so the renderer
can replace its parametric head model with measured data.

Supported inputs
----------------
  * .sofa  -- SOFA "SimpleFreeFieldHRIR" (needs `pip install sofar`)
  * .npz   -- this module's own portable format (see HRTFData.save)
  * .mat   -- generic HRIR mats (CIPIC / KEMAR style keys)
  * arrays -- HRTFData.from_grid(...) / HRTFData.from_arrays(...)

Licences of public datasets differ -- check before shipping:
  CIPIC      : research only
  IRCAM       : research / personal
  SADIE II    : CC BY 4.0 (attribution required, commercial allowed)
  SONICOM     : CC BY 4.0
  MIT KEMAR   : permissive, keep the notice
Always keep the dataset's own attribution.

Coordinate convention
---------------------
azimuth 0 deg = front, +90 deg = right, +/-180 deg = back; elevation 0 = ear
level. SOFA stores azimuth counter-clockwise, so load_sofa() flips the sign by
default (pass azimuth_flip=False to disable).
"""

from __future__ import annotations

import os
from math import gcd

import numpy as np
from scipy.signal import resample_poly
from scipy.spatial import cKDTree

__all__ = [
    "HRTFData",
    "load_hrtf",
    "load_sofa",
    "load_mat",
    "synthetic_hrtf",
]


def _dirs_from_angles(az_deg, el_deg) -> np.ndarray:
    """Unit vectors (x=right, y=front, z=up) from azimuth/elevation degrees."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    x = np.cos(el) * np.sin(az)
    y = np.cos(el) * np.cos(az)
    z = np.sin(el)
    return np.stack([x, y, z], axis=-1)


def _resample_rows(hrirs: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if int(sr_in) == int(sr_out):
        return hrirs
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(hrirs, int(sr_out) // g, int(sr_in) // g, axis=-1)


class HRTFData:
    """A flat set of measured HRIRs with spherical nearest-neighbour lookup."""

    def __init__(self, azimuths, elevations, hrirs_l, hrirs_r, sr,
                 *, interp: str = "idw", k: int = 4, name: str = "hrtf"):
        az = np.asarray(azimuths, dtype=float).ravel()
        el = np.asarray(elevations, dtype=float).ravel()
        hl = np.asarray(hrirs_l, dtype=np.float64)
        hr = np.asarray(hrirs_r, dtype=np.float64)
        if hl.ndim != 2 or hr.ndim != 2:
            raise ValueError("hrirs must be 2D (n_dirs, n_taps)")
        if not (hl.shape[0] == hr.shape[0] == az.size == el.size):
            raise ValueError("azimuth/elevation/HRIR counts do not match")
        self.azimuths = az
        self.elevations = el
        self.hl = hl
        self.hr = hr
        self.sr = int(round(sr))
        self.taps = int(hl.shape[1])
        self.interp = interp
        self.k = int(max(1, k))
        self.name = name
        self.dirs = _dirs_from_angles(az, el)
        self._tree = cKDTree(self.dirs)

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_arrays(cls, azimuths, elevations, hrirs_l, hrirs_r, sr, **kw):
        return cls(azimuths, elevations, hrirs_l, hrirs_r, sr, **kw)

    @classmethod
    def load(cls, path: str, **kw) -> "HRTFData":
        return load_hrtf(path, **kw)

    @classmethod
    def from_grid(cls, azimuths, elevations, hrirs_l, hrirs_r, sr, **kw):
        """Build from a regular grid with HRIRs shaped (n_el, n_az, n_taps)."""
        az = np.asarray(azimuths, dtype=float)
        el = np.asarray(elevations, dtype=float)
        hl = np.asarray(hrirs_l, dtype=np.float64)
        hr = np.asarray(hrirs_r, dtype=np.float64)
        a, e = az.size, el.size
        hl = np.reshape(hl, (e, a, -1))
        hr = np.reshape(hr, (e, a, -1))
        az_grid, el_grid = np.meshgrid(az, el)
        return cls(az_grid.ravel(), el_grid.ravel(),
                   hl.reshape(e * a, -1), hr.reshape(e * a, -1), sr, **kw)

    # ---- queries ------------------------------------------------------
    def hrir(self, theta_deg: float, phi_deg: float):
        """Return interpolated (left, right) HRIR for one direction."""
        q = _dirs_from_angles([theta_deg], [phi_deg])[0]
        kk = min(self.k, len(self.azimuths))
        dist, idx = self._tree.query(q, k=kk)
        dist = np.atleast_1d(dist)
        idx = np.atleast_1d(idx)
        if self.interp == "nearest" or idx.size == 1:
            return self.hl[idx[0]].copy(), self.hr[idx[0]].copy()
        w = 1.0 / (dist + 1e-6)
        w /= w.sum()
        hl = np.tensordot(w, self.hl[idx], axes=(0, 0))
        hr = np.tensordot(w, self.hr[idx], axes=(0, 0))
        return hl, hr

    def resampled(self, target_sr: int) -> "HRTFData":
        if int(target_sr) == self.sr:
            return self
        return HRTFData(self.azimuths, self.elevations,
                        _resample_rows(self.hl, self.sr, target_sr),
                        _resample_rows(self.hr, self.sr, target_sr),
                        int(target_sr), interp=self.interp, k=self.k, name=self.name)

    # ---- io -----------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            azimuths=self.azimuths.astype(np.float32),
            elevations=self.elevations.astype(np.float32),
            hrirs_l=self.hl.astype(np.float32),
            hrirs_r=self.hr.astype(np.float32),
            sr=np.int32(self.sr),
            interp=np.array(self.interp),
            k=np.int32(self.k),
            name=np.array(self.name),
        )

    def __repr__(self) -> str:
        return (f"HRTFData({self.name!r}, dirs={len(self.azimuths)}, "
                f"taps={self.taps}, sr={self.sr}, interp={self.interp})")


def load_hrtf(path: str, **kw) -> HRTFData:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".sofa":
        return load_sofa(path, **kw)
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        interp = str(np.asarray(z["interp"]).item()) if "interp" in z else "idw"
        name = str(np.asarray(z["name"]).item()) if "name" in z else "hrtf"
        params = dict(interp=interp, k=int(np.asarray(z["k"]).item()) if "k" in z else 4,
                      name=name)
        params.update(kw)
        return HRTFData(z["azimuths"], z["elevations"], z["hrirs_l"], z["hrirs_r"],
                        int(np.asarray(z["sr"]).item()), **params)
    if ext == ".mat":
        return load_mat(path, **kw)
    raise ValueError(f"unsupported HRTF file: {path} (use .sofa/.npz/.mat)")


def load_sofa(path: str, *, azimuth_flip: bool = True, name: str | None = None,
              **kw) -> HRTFData:
    try:
        import sofar as sf
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("SOFA loading needs the 'sofar' package: pip install sofar") from exc

    s = sf.read_sofa(path)
    ir = np.asarray(s.Data.IR, dtype=np.float64)
    if ir.ndim == 3 and ir.shape[1] == 2:
        hl, hr = ir[:, 0, :], ir[:, 1, :]
    elif ir.ndim == 3 and ir.shape[1] == 1:
        hl = hr = ir[:, 0, :]
    elif ir.ndim == 2:
        hl = hr = ir
    else:
        raise ValueError("unexpected SOFA Data.IR shape")

    sr = int(round(float(np.asarray(s.Data.SamplingRate).ravel()[0])))

    pos = None
    for attr in ("SourcePosition", "ReceiverPosition"):
        if hasattr(s, attr):
            pos = np.asarray(getattr(s, attr), dtype=float)
            break
    if pos is None:
        raise ValueError("SOFA file has no SourcePosition; cannot recover directions")

    az = pos[:, 0].copy()
    el = pos[:, 1]
    if azimuth_flip:
        az = -az
    return HRTFData(az, el, hl, hr, sr, name=name or os.path.basename(path), **kw)


def load_mat(path: str, *, azimuth_flip: bool = False, name: str | None = None,
             **kw) -> HRTFData:
    from scipy.io import loadmat

    m = loadmat(path)

    def pick(*keys):
        for key in keys:
            if key in m:
                return m[key]
        return None

    hl = pick("hrir_l", "hrirL", "hrirs_l", "left", "hL")
    hr = pick("hrir_r", "hrirR", "hrirs_r", "right", "hR")
    if hl is None or hr is None:
        raise ValueError("mat file needs hrir_l / hrir_r arrays")
    hl = np.asarray(hl, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)

    az = pick("azimuth", "az", "Azimuth", "azimuths")
    el = pick("elevation", "el", "Elevation", "elevations")
    if az is None or el is None:
        raise ValueError("mat file needs azimuth / elevation arrays")
    az = np.asarray(az, dtype=float).ravel()
    el = np.asarray(el, dtype=float).ravel()
    if azimuth_flip:
        az = -az
    sr = kw.pop("sr", 44100)

    # normalise to (n_dirs, n_taps) with matching counts
    taps = hl.shape[-1]
    if hl.ndim == 3:
        if hl.shape[0] == az.size and hl.shape[1] == el.size:      # (az, el, taps)
            hl, hr = hl.transpose(1, 0, 2), hr.transpose(1, 0, 2)   # -> (el, az, taps)
        elif hl.shape[0] == el.size and hl.shape[1] == az.size:    # (el, az, taps)
            pass
        else:
            raise ValueError(f"cannot match HRIR grid {hl.shape} to az={az.size}, el={el.size}")
        return HRTFData.from_grid(az, el, hl, hr, sr, name=name or os.path.basename(path), **kw)

    if hl.ndim == 2:
        n = hl.shape[0]
        if az.size == n and el.size != n:
            el = np.zeros(n)
        elif el.size == n and az.size != n:
            az = np.zeros(n)
        elif az.size != n:
            raise ValueError("2D HRIRs need a per-direction azimuth array")
        return HRTFData(az, el, hl, hr, sr, name=name or os.path.basename(path), **kw)

    raise ValueError("HRIRs must be 2D or 3D")


def synthetic_hrtf(sr: int = 48000, az_step: float = 10.0, el_step: float = 15.0,
                   taps: int = 256, name: str = "synthetic") -> HRTFData:
    """A measured-style HRIR grid generated from the parametric model.

    This is NOT a real dataset -- it exists so the measured-HRTF pipeline can
    be exercised offline. For real work load a .sofa / .mat / .npz instead.
    """
    from .renderer import BinauralConfig, _binaural_transfer  # local import

    az = np.arange(-180.0, 180.0, az_step)
    el = np.arange(-60.0, 60.0 + 1e-9, el_step)
    cfg = BinauralConfig(sr=sr, distance_m=1.0, near_field_boost_db=0.0)
    n = 512
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    hl_all, hr_all = [], []
    for e in el:
        row_l, row_r = [], []
        for a in az:
            h_l, h_r = _binaural_transfer(freqs, np.deg2rad(a), np.deg2rad(e), 1.0, cfg)
            row_l.append(np.fft.irfft(h_l, n=n)[:taps])
            row_r.append(np.fft.irfft(h_r, n=n)[:taps])
        hl_all.append(row_l)
        hr_all.append(row_r)

    return HRTFData.from_grid(az, el, np.array(hl_all), np.array(hr_all), sr,
                              interp="idw", k=4, name=name)


def _main() -> None:  # small converter / inspector CLI
    import argparse

    ap = argparse.ArgumentParser(description="HRTF convert / inspect")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert .sofa/.mat to .npz")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--sr", type=int, default=None, help="resample HRIRs to this rate")
    c.add_argument("--no-flip", action="store_true", help="disable SOFA azimuth flip")

    i = sub.add_parser("info", help="print dataset summary")
    i.add_argument("input")

    s = sub.add_parser("synthetic", help="write a synthetic .npz for testing")
    s.add_argument("output")
    s.add_argument("--sr", type=int, default=48000)

    args = ap.parse_args()
    if args.cmd == "convert":
        data = load_hrtf(args.input, azimuth_flip=not args.no_flip)
        if args.sr:
            data = data.resampled(args.sr)
        data.save(args.output)
        print(f"[ok] {args.input} -> {args.output}  {data!r}")
    elif args.cmd == "info":
        print(load_hrtf(args.input))
    else:
        data = synthetic_hrtf(sr=args.sr)
        data.save(args.output)
        print(f"[ok] wrote synthetic {args.output}  {data!r}")


if __name__ == "__main__":
    _main()
