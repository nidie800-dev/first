"""Independent 360-degree binaural (ASMR) renderer.

Written from scratch with classic psychoacoustics DSP -- no third-party
HRTF database and no model/code from any TTS project is used here.

Pipeline (mono voice in -> headphone binaural out):
  1. STFT of the dry mono signal.
  2. Per-frame binaural transfer function:
       - ITD via the Woodworth spherical-head formula
       - frequency dependent ILD / head-shadow low-pass on the far ear
       - pinna (elevation) notch + front/back high-frequency cue
       - distance gain and near-field (ASMR) low-shelf proximity boost
  3. Optional slow 360-degree orbit of the source around the listener.
  4. Inverse STFT back to the time domain.
  5. Synthetic room: discrete early reflections + decorrelated diffuse
     tail, mixed in as wet signal (small/booth by default for ASMR).
  6. Peak normalisation.

All angles: azimuth 0 deg = front, +90 deg = right, +180/-180 = back.
Elevation: 0 = ear level, +90 = above.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import fftconvolve, istft, resample_poly, stft

__all__ = ["BinauralConfig", "render_binaural", "make_test_signal"]

# room_name -> (rt60 s, pre-delay s, n_early, early_span s, diffuse_gain)
_ROOMS = {
    "dry":   None,
    "booth": (0.20, 0.004, 4,  0.030, 0.90),
    "room":  (0.45, 0.010, 8,  0.060, 1.00),
    "hall":  (1.10, 0.018, 12, 0.100, 1.00),
}


@dataclass
class BinauralConfig:
    sr: int = 48000
    start_azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    distance_m: float = 0.55
    # None -> static source; e.g. 8.0 -> one full orbit every 8 seconds
    rotate_period_s: float | None = None
    orbit_direction: int = 1            # +1 right-to-front, -1 reverse
    room: str = "booth"
    wet: float = 0.18
    n_fft: int = 2048
    hop_ratio: int = 4                  # 75% overlap
    head_radius_m: float = 0.0875
    speed_of_sound: float = 343.0
    near_field_boost_db: float = 2.5
    normalize_peak: float = 0.97
    # optional HRTFData from binaural360.hrtf; when set it replaces the
    # parametric head model with measured HRIRs.
    hrtf: object | None = None


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g)


def _fit(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)))


def _woodworth_itd(alpha: float, a: float, c: float) -> float:
    """Interaural time difference (seconds) for |azimuth| = alpha rad."""
    if alpha <= np.pi / 2.0:
        return (a / c) * (alpha + np.sin(alpha))
    return (a / c) * ((np.pi - alpha) + np.sin(alpha))


def _dist_response(freqs: np.ndarray, dist: float, cfg: BinauralConfig) -> np.ndarray:
    """Distance gain + near-field proximity low shelf (shared by both models)."""
    dist_gain = 0.6 / max(dist, 0.2)
    prox = float(np.clip((0.6 - dist) / 0.6, 0.0, 1.0))
    db = cfg.near_field_boost_db * prox
    bass = 1.0 + (10 ** (db / 20.0) - 1.0) / (1.0 + (freqs / 300.0) ** 2)
    return dist_gain * bass


def _binaural_transfer(freqs: np.ndarray, theta: float, phi: float,
                       dist: float, cfg: BinauralConfig) -> tuple[np.ndarray, np.ndarray]:
    """Parametric complex L/R transfer functions for one source direction."""
    a, c = cfg.head_radius_m, cfg.speed_of_sound
    alpha = abs(theta)

    itd = _woodworth_itd(alpha, a, c)
    shadow = np.sin(alpha)                       # 0..1, no shadow behind

    # far-ear head shadow: one-pole low-pass magnitude
    fc = 18000.0 * np.exp(-2.6 * shadow)
    shadow_mag = 1.0 / np.sqrt(1.0 + (freqs / fc) ** 2)
    far_att = 10 ** (-(8.0 * shadow) / 20.0)     # up to -8 dB broadband

    # pinna notch, centre moves with elevation
    f0 = 7000.0 + 2500.0 * np.sin(phi)
    depth_db = 10.0 * (0.6 if alpha > np.pi / 2 else 1.0)
    bw = 0.28 * f0
    notch = 1.0 - (1.0 - 10 ** (-depth_db / 20.0)) * np.exp(-0.5 * ((freqs - f0) / bw) ** 2)
    if alpha > np.pi / 2:                        # rear: extra HF roll-off
        notch = notch / np.sqrt(1.0 + (freqs / 7000.0) ** 2)

    common = notch * _dist_response(freqs, dist, cfg)

    if theta >= 0:                                # source on the right
        d_r, d_l = 0.0, itd
        g_r, g_l = 1.0, far_att
        sh_r, sh_l = 1.0, shadow_mag
    else:                                         # source on the left
        d_l, d_r = 0.0, itd
        g_l, g_r = 1.0, far_att
        sh_l, sh_r = 1.0, shadow_mag

    hl = common * g_l * sh_l * np.exp(-2j * np.pi * freqs * d_l)
    hr = common * g_r * sh_r * np.exp(-2j * np.pi * freqs * d_r)
    return hl, hr


def _measured_transfer(freqs: np.ndarray, hrtf, theta_deg: float, phi_deg: float,
                       dist: float, cfg: BinauralConfig,
                       n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Complex L/R transfer functions from measured HRIRs at one direction."""
    h_l, h_r = hrtf.hrir(theta_deg, phi_deg)
    hl = np.fft.rfft(h_l, n=n_fft)
    hr = np.fft.rfft(h_r, n=n_fft)
    common = _dist_response(freqs, dist, cfg)
    return hl * common, hr * common


def _make_room_ir(sr: int, preset: str, seed: int) -> np.ndarray | None:
    """Synthetic room impulse response (early taps + diffuse tail)."""
    p = _ROOMS.get(preset)
    if p is None:
        return None
    rt60, predelay, n_early, early_span, diffuse = p
    rng = np.random.default_rng(seed)
    n = int(sr * (predelay + rt60 * 1.2)) + 1
    ir = np.zeros(n)
    pd = int(predelay * sr)

    for k in range(n_early):
        frac = (k + 1) / n_early
        d = pd + int(sr * (0.004 + early_span * frac * rng.uniform(0.6, 1.0)))
        if d < n:
            sign = 1.0 if rng.random() > 0.5 else -1.0
            ir[d] += sign * 0.7 * frac ** 0.5 * rng.uniform(0.5, 1.0)

    t = np.arange(n) / sr
    tau = rt60 / 6.9
    env = np.exp(-np.maximum(t - predelay, 0.0) / tau)
    env[:pd] = 0.0
    tail = rng.standard_normal(n) * env
    tail = np.convolve(tail, np.ones(8) / 8, mode="same")
    ir += diffuse * tail

    e = np.sqrt(np.sum(ir ** 2))
    if e > 0:
        ir /= e
    return ir


def render_binaural(audio: np.ndarray, cfg: BinauralConfig | None = None,
                    sr: int | None = None) -> np.ndarray:
    """Render mono (or stereo) audio into a binaural stereo signal.

    Returns float32 array of shape (n_samples, 2).
    """
    cfg = cfg or BinauralConfig()
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.asarray(x).reshape(-1)

    sr_in = int(sr) if sr is not None else cfg.sr
    x = _resample(x, sr_in, cfg.sr)
    sr = cfg.sr
    n = len(x)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)

    n_fft = cfg.n_fft
    hrtf = cfg.hrtf
    if hrtf is not None:
        if int(getattr(hrtf, "sr", sr)) != sr:
            hrtf = hrtf.resampled(sr)
        taps = int(getattr(hrtf, "taps", 0))
        if taps and n_fft < taps:
            n_fft = 1 << int(np.ceil(np.log2(taps)))
    hop = max(1, n_fft // cfg.hop_ratio)
    noverlap = n_fft - hop

    freqs, times, z = stft(x, fs=sr, nperseg=n_fft, noverlap=noverlap,
                           window="hann", boundary="zeros", padded=True)

    if cfg.rotate_period_s and cfg.rotate_period_s > 0:
        az = cfg.start_azimuth_deg + cfg.orbit_direction * 360.0 * (times / cfg.rotate_period_s)
    else:
        az = np.full_like(times, cfg.start_azimuth_deg)
    az_deg = ((az + 180.0) % 360.0) - 180.0
    theta = np.deg2rad(az_deg)
    phi = np.deg2rad(cfg.elevation_deg)

    yl = np.empty_like(z)
    yr = np.empty_like(z)
    for i in range(z.shape[1]):
        if hrtf is not None:
            hl, hr = _measured_transfer(freqs, hrtf, float(az_deg[i]),
                                        cfg.elevation_deg, cfg.distance_m, cfg, n_fft)
        else:
            hl, hr = _binaural_transfer(freqs, float(theta[i]), phi, cfg.distance_m, cfg)
        yl[:, i] = z[:, i] * hl
        yr[:, i] = z[:, i] * hr

    _, left = istft(yl, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann", boundary=True)
    _, right = istft(yr, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann", boundary=True)
    left = _fit(np.asarray(left), n)
    right = _fit(np.asarray(right), n)

    if cfg.room != "dry" and cfg.wet > 0:
        ir_l = _make_room_ir(sr, cfg.room, seed=1)
        ir_r = _make_room_ir(sr, cfg.room, seed=2)
        if ir_l is not None and ir_r is not None:
            wl = fftconvolve(left, ir_l)[:n]
            wr = fftconvolve(right, ir_r)[:n]
            left = (1.0 - cfg.wet) * left + cfg.wet * wl
            right = (1.0 - cfg.wet) * right + cfg.wet * wr
            fade = min(int(0.010 * sr), n)          # avoid end click
            if fade > 0:
                ramp = np.linspace(1.0, 0.0, fade)
                left[-fade:] *= ramp
                right[-fade:] *= ramp

    out = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out * (cfg.normalize_peak / peak)
    return out.astype(np.float32)


def make_test_signal(duration_s: float = 6.0, sr: int = 48000,
                     seed: int = 0) -> np.ndarray:
    """Synthetic 'vocal-ish' test signal for trying the renderer."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr
    noise = rng.standard_normal(len(t))

    # two broad formants to make it voice-like rather than pure noise
    sig = np.zeros_like(noise)
    for f0, bw in [(500.0, 120.0), (1500.0, 250.0), (2800.0, 400.0)]:
        sig += _bandpass(noise, f0, bw, sr)
    sig /= (np.max(np.abs(sig)) + 1e-9)

    # slow syllabic amplitude envelope (whisper-like)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
    env *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t) ** 2
    return (sig * env * 0.7).astype(np.float64)


def _bandpass(x: np.ndarray, f0: float, bw: float, sr: int) -> np.ndarray:
    """Very small optional helper: one biquad band-pass via FFT mask."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec = np.fft.rfft(x)
    mask = np.exp(-0.5 * ((freqs - f0) / bw) ** 2)
    return np.fft.irfft(spec * mask, n=n)
