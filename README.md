# Tropical Downpour on Corrugated Iron - Texture Dataset

**Course:** Generative Algorithms for Sound and Music  
**Assignment 1:** Texture Dataset Curation  

---

## Concept

For this assignment I wanted a sound texture that would be trivially realistic to synthesise. After an initial attempt at singing sand dunes (which ended up sounding like an FM synth no matter what I did), I pivoted to something whose source material is literally noise: **rain**.

Heavy rain is a Poisson process, millions of independent drop impacts per second whose superposition converges to coloured Gaussian noise by the central limit theorem. That means filtered white noise *is* physically correct, not an approximation. The corrugated iron surface adds strong resonant colouring (narrow-band modal peaks) that gives the texture its distinctive metallic character.

The result is a continuously evolving texture that is 100 % noise-based, easy to parameterise, and immediately recognisable. Two parameters control the sound:

- **Rainfall rate** — how many drops per second hit the surface
- **Drop diameter** — the median size of the drops, which shifts the spectral balance

These interact in interesting, non-trivial ways: large drops at low intensity sound like a slow, deep drumming; small drops at high intensity sound like aggressive static; and everything in between produces a rich, immersive downpour.

## Parameters

| Parameter | Unit | Range | Perceptual effect |
|-----------|------|-------|-------------------|
| `rainfall_rate` | mm/h | 2 – 120 | Overall loudness and density. Light drizzle → moderate rain → torrential downpour. Higher values also bring in splash/runoff (low-freq) and more metallic resonance excitation. |
| `drop_diameter` | mm | 0.5 – 5.0 | Spectral tilt. Small drops (0.5 mm) produce bright, hissy texture concentrated above 3 kHz. Large drops (5 mm) shift energy into the low-mids (200–1500 Hz) and excite the metal resonances more strongly. |

Both parameters are stored in their physical units in the CSV files. Normalisation to [0, 1] is done downstream using the `min` / `max` from `parameters.json`.

## Synthesis approach

Everything is generated in Python with `numpy` and `scipy` — no samples, no ML, no external audio libraries. The core idea is subtractive synthesis on multiple decorrelated white-noise sources, shaped by time-varying gain curves.

### Layers

1. **Rain body** — A 6-band filterbank (120 Hz – 11 kHz) applied to a single white-noise source. The `drop_diameter` parameter crossfades between bands using Gaussian weighting: small drops activate the upper bands, large drops activate the lower bands. Amplitude scales with `rainfall_rate` using a perceptual power curve (~0.65 exponent, since perceived loudness isn't linear with drop count).

2. **Corrugated iron resonance** — Five narrow bandpass filters at 1180, 2740, 4410, 6200, and 8350 Hz (typical modal frequencies of 0.5 mm galvanised steel roofing) applied to an *independent* noise source. Larger drops excite the resonances more strongly. This layer gives the sound its characteristic metallic "ring."

3. **Splash / runoff** — Low-passed noise (< 200 Hz) that gates on above ~20 % intensity, representing accumulated water and turbulent surface flow. Modulated by its own slow Ornstein-Uhlenbeck process (τ ≈ 14 s) to mimic natural flow variation.

4. **HF spatter** — Bandpassed 5–11.5 kHz noise representing fine secondary droplets and splash mist. More prominent for small drops and high intensity.

### Stochastic modulation

This is what makes it sound alive instead of static:

- **Wind gusts** — Two independent [Ornstein-Uhlenbeck processes](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) (mean-reverting random walks) at different time-scales:
  - Slow gusts (τ ≈ 8 s) — creates large amplitude swells, like wind pushing the rain in waves
  - Micro-gusts (τ ≈ 1.5 s) — rapid subtle fluttering within the larger envelope
  
  Their product modulates the rain body + metal layers, producing natural, non-periodic loudness variation.

- **Spectral drift** — A third OU process (τ ≈ 5 s) adds gentle spectral wander independent of the control parameter, simulating how wind shifts the effective drop-size distribution moment to moment.

The OU processes are implemented as AR(1) IIR filters on white noise via `scipy.signal.lfilter`, so they run at full sample-rate without any Python loops.

## Dataset structure

```
dataset-creation-cantus-machina/
├── README.md                    # This file
├── generate_rain.py             # Synthesis script
└── raw/                         # Raw dataset (Option 1: simple structure)
    ├── parameters.json          # Parameter specification
    ├── rain_metal_01.wav        # Audio file 1 (3 min)
    ├── rain_metal_01.csv        # Annotations 1 (13 500 frames)
    ├── rain_metal_02.wav        # Audio file 2
    ├── rain_metal_02.csv        # Annotations 2
    ├── rain_metal_03.wav        # Audio file 3
    ├── rain_metal_03.csv        # Annotations 3
    ├── rain_metal_04.wav        # Audio file 4
    └── rain_metal_04.csv        # Annotations 4
```

## Parameter coverage

Each file uses a different sweep pattern so the 4 files together cover the 2D parameter space without large gaps:

| File | `rainfall_rate` | `drop_diameter` |
|------|----------------|-----------------|
| 01 | Linear ramp 0 → 1 | Sine arch (peaks mid-file) |
| 02 | 2-cycle sinusoid | Linear ramp 1 → 0 |
| 03 | 3-cycle sinusoid | 2-cycle cosine (phase-shifted) |
| 04 | Linear ramp 1 → 0 | 4-cycle sinusoid |

## Technical specs

| Property | Value |
|----------|-------|
| Sample rate | 24 000 Hz |
| Bit depth | 32-bit float |
| Format | WAV |
| Total duration | 12 min (4 × 3 min) |
| Annotation rate | 75 fps |
| Frames per file | 13 500 |
| Total frames | 54 000 |

## How to regenerate

```bash
pip install numpy scipy pandas    # if not already installed
python generate_rain.py
```

Takes ~15 seconds. Outputs go into `raw/`. The random seed is not fixed so each run produces slightly different stochastic textures (the deterministic parameter sweeps are identical across runs).