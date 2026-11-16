"""A Japanese CV voicebank synthesised from scratch - no recordings, no TTS, no voice model.

Adachi Rei's original bank is described as built from generated sounds (sine waves and noise) edited by hand, not sampled
from a speaker. This does the same with a small source-filter synthesiser:

    glottal pulse train (constant pitch) + aspiration noise  ->  cascade of 5 formant resonators (time-varying)
    + separate band-passed noise for fricatives/bursts       ->  one clip per hiragana CV unit of `morae.MORAE`

Vowels are steady (so the editor can stretch them cleanly); consonants are rule-based (stop bursts + aspiration, fricative
noise bands, nasal murmur, tap, glides) with formant transitions into the vowel. Every clip is at exactly F0, and gets a
flat .frq file. The voice is intentionally synthetic and buzzy.

    python -m voice2utau.synthbank -o out --name "Synth CV"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from . import audio
from .bank import write_bank
from .extract import Clip
from .morae import MORAE, Mora
from .sfxbank import write_frq

SR = audio.SR_BANK
F0 = 261.63                       # every sample is exactly C4
BLOCK = 64
VOWEL_MS = 380                    # steady vowel after the transition
TRANS_MS = 55
# Japanese vowels, female-ish scale: (F1, F2, F3)
VOWELS = {"a": (850, 1330, 2800), "i": (320, 2650, 3300), "u": (360, 1450, 2500), "e": (520, 2150, 2900),
          "o": (500, 900, 2600)}
F4, F5 = 3800, 4700
BW = np.array([90.0, 110.0, 160.0, 220.0, 300.0])

PAL = (300, 2400, 3000)           # palatal locus ("i"-like)
LAB = (330, 800, 2200)            # labial locus
ALV = (350, 1750, 2700)           # alveolar locus
VEL = (350, 1500, 2400)           # velar locus
# class -> (kind, consonant length ms, locus, extra)
# kinds: stop_v (voiceless stop), stop_d (voiced stop), fric, fric_v, aff, aff_v, nasal, tap, glide, h, vowel
CLASSES: dict[str, tuple] = {
    "": ("vowel", 20, None, {}),
    "k": ("stop_v", 62, VEL, {"burst": 2600, "bw": 1800, "vot": 42}),
    "ky": ("stop_v", 70, PAL, {"burst": 3600, "bw": 2000, "vot": 48}),
    "t": ("stop_v", 55, ALV, {"burst": 4600, "bw": 2600, "vot": 36}),
    "p": ("stop_v", 52, LAB, {"burst": 1300, "bw": 1500, "vot": 28}),
    "py": ("stop_v", 60, PAL, {"burst": 1800, "bw": 1800, "vot": 34}),
    "g": ("stop_d", 42, VEL, {"burst": 2400, "bw": 1800}),
    "gy": ("stop_d", 48, PAL, {"burst": 3400, "bw": 2000}),
    "d": ("stop_d", 38, ALV, {"burst": 4200, "bw": 2600}),
    "b": ("stop_d", 36, LAB, {"burst": 1200, "bw": 1500}),
    "by": ("stop_d", 44, PAL, {"burst": 1700, "bw": 1800}),
    "s": ("fric", 105, ALV, {"cf": 7600, "bw": 3200, "hp": 4200, "amp": 0.62}),
    "sh": ("fric", 115, PAL, {"cf": 3600, "bw": 1700, "hp": 2000, "amp": 1.0}),
    "z": ("fric_v", 70, ALV, {"cf": 7200, "bw": 3200, "hp": 4200, "amp": 0.35}),
    "j": ("aff_v", 85, PAL, {"burst": 3600, "bw": 2000, "cf": 3400, "fbw": 1700, "hp": 2000, "amp": 0.4}),
    "ch": ("aff", 95, PAL, {"burst": 3800, "bw": 2000, "cf": 3600, "fbw": 1700, "hp": 2000, "amp": 0.9}),
    "ts": ("aff", 90, ALV, {"burst": 5200, "bw": 2600, "cf": 7400, "fbw": 3200, "hp": 4200, "amp": 0.2}),
    "h": ("h", 85, None, {}),
    "hy": ("h", 95, PAL, {"gain": 2.6}),
    "f": ("fric", 85, LAB, {"cf": 2800, "bw": 4200, "hp": 800, "amp": 0.075}),
    "n": ("nasal", 62, ALV, {"f2": 1500}),
    "ny": ("nasal", 70, PAL, {"f2": 2300}),
    "m": ("nasal", 62, LAB, {"f2": 1000}),
    "my": ("nasal", 72, PAL, {"f2": 1900}),
    "r": ("tap", 34, ALV, {}),
    "ry": ("tap", 48, PAL, {}),
    "y": ("glide", 48, PAL, {}),
    "w": ("glide", 55, (320, 760, 2300), {}),
}


def _db(x: float) -> float:
    return 10 ** (x / 20)


def _glottal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Differentiated Rosenberg pulse train at exactly F0 (tiny shimmer so it is not a pure comb)."""
    ph = (np.arange(n) * F0 / SR) % 1.0
    tp, tn = 0.5, 0.1
    g = np.where(ph < tp, 0.5 * (1 - np.cos(np.pi * ph / tp)),
                 np.where(ph < tp + tn, np.cos(0.5 * np.pi * (ph - tp) / tn), 0.0))
    d = np.diff(g, prepend=0.0)
    return d / max(1e-9, np.abs(d).max())


def _ramp(n: int, a: int, b: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Piecewise-linear ramp from lo (before a) to hi (after b)."""
    return np.interp(np.arange(n), [a, max(b, a + 1)], [lo, hi])


def _track(loc: float, tgt: float, n: int, c: int, tr: int) -> np.ndarray:
    return np.interp(np.arange(n), [0, c, c + tr, n], [loc, loc, tgt, tgt])


def _cascade(src: np.ndarray, freqs: list[np.ndarray], bwscale: np.ndarray | None = None) -> np.ndarray:
    """Time-varying cascade of resonators (Klatt form), updated every BLOCK samples."""
    n = len(src)
    y = src.astype("float64").copy()
    for k, f in enumerate(freqs):
        zi = np.zeros(2)
        out = np.empty(n)
        for s in range(0, n, BLOCK):
            e = min(n, s + BLOCK)
            mid = min(n - 1, (s + e) // 2)
            bw = BW[k] * (bwscale[mid] if bwscale is not None and k in (1, 2) else 1.0)
            fc = float(f[mid])
            c = -np.exp(-2 * np.pi * bw / SR)
            b = 2 * np.exp(-np.pi * bw / SR) * np.cos(2 * np.pi * fc / SR)
            out[s:e], zi = lfilter([1 - b - c], [1.0, -b, -c], y[s:e], zi=zi)
        y = out
    return y


def _band_noise(n: int, cf: float, bw: float, hp: float, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n)
    lo, hi = max(200.0, cf - bw), min(SR / 2 - 500, cf + bw)
    x = sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)
    x = sosfilt(butter(2, hp, "high", fs=SR, output="sos"), x)
    return x / max(1e-9, np.abs(x).max())


def _burst(n: int, at: int, ms: float, cf: float, bw: float, rng: np.random.Generator) -> np.ndarray:
    m = int(ms / 1000 * SR)
    x = np.zeros(n)
    seg = _band_noise(m, cf, bw, 500.0, rng) * np.exp(-np.arange(m) / (m / 3.0))
    x[at:at + m] = seg[: max(0, n - at)]
    return x


def synth(m: Mora, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Render one CV mora. Returns (audio at 44.1 kHz, consonant length in seconds)."""
    rng = rng or np.random.default_rng(1)
    if m.key == "nn":
        return _syllabic_n(rng)
    kind, c_ms, locus, ex = CLASSES[m.cls]
    c, tr = int(c_ms / 1000 * SR), int(TRANS_MS / 1000 * SR)
    n = c + tr + int(VOWEL_MS / 1000 * SR)
    v = VOWELS[m.vowel]
    loc = locus or v
    if kind == "h":
        loc = locus or v                                           # h: aspiration shaped by the vowel itself
    f = [_track(loc[0], v[0], n, c, tr), _track(loc[1], v[1], n, c, tr), _track(loc[2], v[2], n, c, tr),
         np.full(n, F4), np.full(n, F5)]
    glot = _glottal(n, rng)
    voiced = np.zeros(n)
    asp = np.zeros(n)
    fric = np.zeros(n)
    bwscale = None
    on = int(0.006 * SR)
    if kind == "vowel":
        voiced = _ramp(n, 0, on)
    elif kind == "stop_v":
        voiced = _ramp(n, c, c + on)
        fric += _burst(n, 0, 11, ex["burst"], ex["bw"], rng) * 2.2
        a = int(0.011 * SR)
        asp = np.where((np.arange(n) >= a) & (np.arange(n) < c + on), 1.0, 0.0) * 1.3 * _ramp(n, a, a + on)
    elif kind == "stop_d":
        pv = int(0.020 * SR)
        voiced = np.where(np.arange(n) < pv, 0.18, 1.0) * _ramp(n, 0, on)
        voiced[pv: c] = 0.18 + 0.4 * (np.arange(c - pv) / max(1, c - pv))
        voiced[c:] = 1.0
        fric += _burst(n, pv, 7, ex["burst"], ex["bw"], rng) * 0.45
    elif kind in ("fric", "fric_v", "aff", "aff_v"):
        fr0 = int(0.012 * SR) if kind.startswith("aff") else 0
        cf, fbw = ex.get("cf"), ex.get("fbw", ex.get("bw"))
        noise = _band_noise(n, cf, fbw, ex["hp"], rng)
        end = c + int(0.012 * SR)
        env = _ramp(n, fr0, fr0 + int(0.015 * SR)) * (1 - _ramp(n, c - int(0.01 * SR), end))
        fric += noise * env * ex["amp"]
        if kind.startswith("aff"):
            fric += _burst(n, 0, 9, ex["burst"], ex["bw"], rng) * 0.7
        if kind.endswith("_v"):
            voiced = np.minimum(_ramp(n, 0, int(0.01 * SR)), 1.0) * np.where(np.arange(n) < c, 0.45, 1.0)
        else:
            voiced = _ramp(n, c - int(0.005 * SR), c + on)
        asp = 0.18 * np.where(np.arange(n) < c, 1.0, 0.0)
    elif kind == "h":
        a = int(0.02 * SR)
        asp = _ramp(n, 0, a) * (1 - _ramp(n, c - a, c + int(0.01 * SR))) * 0.8 * ex.get("gain", 1.0)
        voiced = _ramp(n, c - int(0.015 * SR), c + int(0.01 * SR))
    elif kind == "nasal":
        nas = (260, ex["f2"], 2800)
        f = [_track(nas[0], v[0], n, c, tr), _track(nas[1], v[1], n, c, tr), _track(nas[2], v[2], n, c, tr),
             np.full(n, F4), np.full(n, F5)]
        voiced = np.where(np.arange(n) < c, 0.55, 1.0) * _ramp(n, 0, on)
        bwscale = np.where(np.arange(n) < c, 2.6, 1.0)                 # damped F2/F3 in the murmur
    elif kind == "tap":
        t = np.arange(n) / SR
        dip = 1.0 - 0.85 * np.exp(-(((t - 0.018) / 0.009) ** 2))
        voiced = _ramp(n, 0, on) * dip
    elif kind == "glide":
        voiced = _ramp(n, 0, int(0.015 * SR))
    glot_src = glot * voiced + rng.standard_normal(n) * asp * 0.35
    y = _cascade(glot_src, f, bwscale) + fric * 2.2
    # level the vowel by its first formant (open vowels are louder) and shape the end
    tail = int(0.035 * SR)
    y[-tail:] *= np.linspace(1, 0, tail)
    y = y / max(1e-9, np.abs(y).max()) * 0.8
    return y.astype("float32"), c / SR


def _syllabic_n(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    n = int(0.42 * SR)
    f = [np.full(n, 260.0), np.full(n, 1150.0), np.full(n, 2600.0), np.full(n, F4), np.full(n, F5)]
    voiced = _ramp(n, 0, int(0.01 * SR)) * 0.8
    y = _cascade(_glottal(n, rng) * voiced, f, np.full(n, 2.4))
    tail = int(0.05 * SR)
    y[-tail:] *= np.linspace(1, 0, tail)
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype("float32"), 0.02


def build(out_dir: Path, name: str = "Synth CV") -> Path:
    rng = np.random.default_rng(7)
    clips: dict[str, Clip] = {}
    for key, m in MORAE.items():
        y, split = synth(m, rng)
        y = audio.normalise_level(y)
        clips[key] = Clip(m, y, split, F0, "synth", 1.0, "klatt-style source-filter")
    report = {"kind": "synthbank", "f0_hz": F0, "units": len(clips),
              "samples": [{"key": k, "source": "synth", "f0": F0} for k in clips],
              "method": "glottal pulse train + aspiration through a 5-formant cascade; band-passed noise for fricatives/bursts"}
    root = write_bank(clips, out_dir, name, report)
    for k in clips:
        write_frq(root / f"{k}_wav.frq", len(clips[k].audio))
    (root / "readme.txt").write_text(
        f"{name}\nA Japanese hiragana CV voicebank synthesised from scratch (source-filter synthesis, constant C4 = {F0:.2f} Hz), "
        "no recordings and no voice model.\nSet the aliases to hiragana. Vowels are steady so long notes stretch cleanly; the voice "
        "is intentionally synthetic.\nGenerated by voice2utau.synthbank - free to use.\n", encoding="utf-8")
    return root


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--name", default="Synth CV")
    a = ap.parse_args()
    root = build(a.out, a.name)
    print(f"wrote {len(MORAE)} units to {root}")


if __name__ == "__main__":
    main()
