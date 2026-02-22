#!/usr/bin/env python3
"""
Tropical Downpour on Corrugated Iron — Texture Dataset Generator
=====================================================================
Synthesises a realistic rain-on-metal texture controlled by two
continuous parameters:

    rainfall_rate   2 – 120 mm/h    intensity / density of the rain
    drop_diameter   0.5 – 5.0 mm    median drop size

Acoustic model
--------------
Rain at moderate-to-heavy intensity is a naturally stochastic process
whose superposition converges to coloured noise (central limit theorem).
The spectral envelope depends on drop-size distribution, and the
metallic surface imprints resonant modes on the sound.

Four decorrelated noise layers:
  1. Rain body        — 6-band filterbank crossfaded by drop_diameter
  2. Metal resonance  — narrow peaks at corrugated-iron modal frequencies
  3. Splash / runoff  — LF noise gated by intensity (water accumulation)
  4. HF spatter       — high-freq sparkle, louder for small drops

All layers are modulated by independent Ornstein-Uhlenbeck processes
(mean-reverting stochastic walks) to produce organic, non-periodic
fluctuation that sounds like real wind gusts.

Output
------
4 × 3-minute WAV files  (12 min total, 24 kHz, 32-bit float)
4 matching CSV files     (13 500 rows @ 75 fps, physical-unit columns)
1 parameters.json
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, lfilter
import pandas as pd
import json
import os

# ── Configuration ────────────────────────────────────────────────────────────
SR      = 24_000          # sample-rate (model requirement)
FPS     = 75              # annotation frame-rate
N_FILES = 4               # number of audio clips
DUR_S   = 180             # seconds per clip  →  4 × 3 min = 12 min
OUT_DIR = "raw"

# physical parameter ranges
RATE_MIN,  RATE_MAX  =   2.0, 120.0   # mm / h  (rainfall rate)
DROP_MIN,  DROP_MAX  =   0.5,   5.0   # mm      (median drop diameter)


# ── DSP primitives ───────────────────────────────────────────────────────────

def _bp(sig, lo, hi, order=2):
    """Butterworth band-pass (safe against edge frequencies)."""
    nyq  = SR / 2.0
    lo_n = np.clip(lo / nyq, 1e-4, 0.9999)
    hi_n = np.clip(hi / nyq, lo_n + 1e-4, 0.9999)
    sos  = butter(order, [lo_n, hi_n], btype="band", output="sos")
    return sosfilt(sos, sig)


def _lp(sig, fc, order=2):
    """Butterworth low-pass."""
    return sosfilt(
        butter(order, min(fc / (SR / 2), 0.9999), "low", output="sos"), sig
    )


def _hp(sig, fc, order=2):
    """Butterworth high-pass."""
    return sosfilt(
        butter(order, max(fc / (SR / 2), 1e-4), "high", output="sos"), sig
    )


def _ou(n, tau_s, sigma=1.0):
    """
    Ornstein-Uhlenbeck process (mean-reverting stochastic walk).

    Implemented as an AR(1) IIR filter on white noise — fully vectorised,
    runs at sample-rate without a Python loop.

    Parameters
    ----------
    n      : number of samples
    tau_s  : correlation time in seconds (larger → slower drift)
    sigma  : asymptotic standard deviation
    """
    alpha = 1.0 / (tau_s * SR)
    drive = sigma * np.sqrt(2.0 * alpha) * np.random.randn(n)
    return lfilter([1.0], [1.0, -(1.0 - alpha)], drive)


# ── Parameter curves ─────────────────────────────────────────────────────────

def _make_curves(dur, file_idx):
    """
    Generate deterministic sweep curves (normalised 0–1) for each file.
    Four complementary patterns ensure good coverage of the 2-D space.
    """
    n = int(dur * SR)
    t = np.linspace(0, dur, n)

    patterns = [
        # 0: rate ramps up, drop sine arch
        (t / dur,
         0.5 + 0.45 * np.sin(2 * np.pi * t / dur)),
        # 1: rate 2-cycle sine, drop ramps down
        (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t / dur - np.pi / 2),
         1.0 - t / dur),
        # 2: rate 3-cycle sine, drop 2-cycle cosine (phase-shifted)
        (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t / dur),
         0.5 + 0.45 * np.cos(2 * np.pi * 2 * t / dur + 0.8)),
        # 3: rate ramps down, drop 4-cycle sine
        (1.0 - t / dur,
         0.5 + 0.4 * np.sin(2 * np.pi * 4 * t / dur)),
    ]

    rn, dn = patterns[file_idx % len(patterns)]
    rn = np.clip(rn, 0.0, 1.0)
    dn = np.clip(dn, 0.0, 1.0)

    rate = RATE_MIN + rn * (RATE_MAX - RATE_MIN)
    drop = DROP_MIN + dn * (DROP_MAX - DROP_MIN)
    return rate, drop, rn, dn


# ── Synthesis engine ─────────────────────────────────────────────────────────

def _synthesise(rn, dn):
    """
    Build the rain texture from four decorrelated noise layers.

    Parameters
    ----------
    rn : ndarray[float]   normalised rainfall rate   [0, 1]
    dn : ndarray[float]   normalised drop diameter   [0, 1]

    Returns
    -------
    audio : ndarray[float32]   mono waveform, peak-normalised
    """
    N = len(rn)

    # ── Layer 1 — Rain body (filterbank spectral crossfade) ──────────────
    #
    # Six overlapping bands span 120 Hz – 11 kHz.  Drop-size selects which
    # bands dominate: small drops (dn→0) activate high bands, large drops
    # (dn→1) push energy into the lows.  This is physically correct — the
    # spectrum of rainfall noise is governed by drop-size distribution
    # (Marshall & Palmer, 1948).

    w1 = np.random.randn(N)

    band_edges = [
        ( 120,  450),   # sub-bass thump
        ( 350,  900),   # low body
        ( 750, 2100),   # low-mid punch
        (1700, 4200),   # presence
        (3400, 7200),   # brilliance
        (5800, 11000),  # air / sizzle
    ]
    bands = [_bp(w1, lo, hi) for lo, hi in band_edges]

    nb   = len(bands)
    rain = np.zeros(N)
    for i, b in enumerate(bands):
        pos    = i / (nb - 1)               # 0 = lowest … 1 = highest
        target = 1.0 - dn                   # small drop → high target
        wt     = np.exp(-((target - pos) ** 2) / (2 * 0.20 ** 2))
        rain  += wt * b

    rain *= (0.04 + rn ** 0.65) * 0.58

    # ── Layer 2 — Corrugated-iron resonance ──────────────────────────────
    #
    # Corrugated galvanised steel (0.5 mm typical roofing) has distinct
    # modal resonances.  These are modelled as narrow bandpass filters on
    # an independent noise source, so the rain excites the surface modes
    # without perfect correlation to the body layer.

    w2 = np.random.randn(N)

    modes = [
        (1180,  55),    # first mode
        (2740,  75),
        (4410,  95),
        (6200, 110),
        (8350, 140),    # fifth mode
    ]
    metal = np.zeros(N)
    for cf, bw in modes:
        filt   = _bp(w2, cf - bw, cf + bw, order=3)
        level  = 1.0 / (1.0 + (cf / 2000.0) ** 0.4)
        metal += filt * level

    # larger drops excite surface modes more strongly
    metal *= rn ** 0.5 * (0.20 + 0.80 * dn) * 0.20

    # ── Layer 3 — Splash / water runoff ──────────────────────────────────
    #
    # At higher rainfall rates, water accumulates and its turbulent flow
    # adds low-frequency energy.  The layer fades in above ~20 % intensity
    # and wanders slowly via its own OU modulator.

    w3     = np.random.randn(N)
    splash = _lp(w3, 200) + _bp(w3, 80, 350) * 0.4

    gate     = np.clip((rn - 0.20) / 0.80, 0.0, 1.0) ** 1.6
    flow_mod = 0.55 + 0.45 * np.tanh(_ou(N, tau_s=14.0, sigma=0.40))

    splash *= gate * flow_mod * 0.13

    # ── Layer 4 — High-frequency spatter ─────────────────────────────────
    #
    # Fine droplets and secondary splash create a delicate HF shimmer that
    # is more prominent for small drops and high intensity.

    w4      = np.random.randn(N)
    spatter = _bp(w4, 5000, 11500)
    spatter *= rn ** 0.75 * (0.55 + 0.45 * (1.0 - dn)) * 0.09

    # ── Stochastic wind-gust modulation ──────────────────────────────────
    #
    # Two independent OU processes at different time-scales:
    #   • slow gusts  (τ ≈ 8 s)   — large swells in loudness
    #   • micro-gusts (τ ≈ 1.5 s) — rapid, subtle fluttering
    # Their product gives a natural, non-periodic envelope.

    gust  = 0.60 + 0.40 * np.tanh(_ou(N, tau_s=8.0,  sigma=0.38))
    micro = 0.82 + 0.18 * np.tanh(_ou(N, tau_s=1.5,  sigma=0.20))
    env   = gust * micro

    # ── Natural spectral drift ───────────────────────────────────────────
    #
    # Wind gusts also shift the effective drop-size distribution slightly,
    # causing gentle spectral wander independent of the control parameter.

    w5          = np.random.randn(N)
    drift       = _ou(N, tau_s=5.0, sigma=0.06)
    drift_layer = _bp(w5, 600, 4500) * drift * rn * 0.08

    # ── Final mix & post-processing ──────────────────────────────────────

    mix = (rain + metal + drift_layer) * env + splash + spatter

    # soft saturation — prevents clipping while adding gentle warmth
    mix = np.tanh(mix * 1.25)

    # fade in / out (0.4 s)
    fade = int(0.4 * SR)
    mix[:fade]  *= np.linspace(0.0, 1.0, fade)
    mix[-fade:] *= np.linspace(1.0, 0.0, fade)

    # peak-normalise to -0.9 dBFS
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix *= 0.90 / peak

    return mix.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 64)
    print("Tropical Downpour on Corrugated Iron — Generator")
    print("=" * 64)

    for i in range(N_FILES):
        tag = f"rain_metal_{i + 1:02d}"
        print(f"\n▶ [{i + 1}/{N_FILES}]  {tag}  ({DUR_S} s)")

        # deterministic parameter curves
        rate, drop, rn, dn = _make_curves(DUR_S, i)

        # synthesise
        print("  … synthesising layers")
        audio = _synthesise(rn, dn)

        # write WAV
        wav_path = os.path.join(OUT_DIR, f"{tag}.wav")
        wavfile.write(wav_path, SR, audio)
        print(f"  ✔ {tag}.wav   ({len(audio) / SR:.0f} s, {SR} Hz)")

        # write CSV @ 75 fps
        n_frames = DUR_S * FPS
        idx = np.linspace(0, len(rate) - 1, n_frames).astype(int)
        df  = pd.DataFrame({
            "rainfall_rate": rate[idx],
            "drop_diameter": drop[idx],
        })
        csv_path = os.path.join(OUT_DIR, f"{tag}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✔ {tag}.csv   ({n_frames} frames @ {FPS} fps)")

    # ── parameters.json ──────────────────────────────────────────────────
    params = {
        "parameter_1": {
            "name":  "rainfall_rate",
            "type":  "continuous",
            "unit":  "mm/h",
            "min":   RATE_MIN,
            "max":   RATE_MAX,
        },
        "parameter_2": {
            "name":  "drop_diameter",
            "type":  "continuous",
            "unit":  "mm",
            "min":   DROP_MIN,
            "max":   DROP_MAX,
        },
    }
    params_path = os.path.join(OUT_DIR, "parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"\n  ✔ parameters.json")

    total_min = N_FILES * DUR_S / 60.0
    print(f"\n{'=' * 64}")
    print(f"✅  Done — {N_FILES} files, {total_min:.0f} minutes total")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
