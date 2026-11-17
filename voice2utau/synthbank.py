"""A Japanese CV voicebank synthesised from scratch - no recordings, no TTS, no voice model.

Adachi Rei's original bank is described as built from generated sounds (sine waves and noise) edited by hand, not sampled
from a speaker. This does the same with a small source-filter (Klatt-style) synthesiser:

    glottal pulses (constant mean pitch, slight jitter/shimmer) + aspiration noise
        -> cascade of 5 time-varying formant resonators (+ a nasal anti-resonance for m/n/ny)
    + separate band-passed noise for fricatives and stop bursts
        -> one clip per hiragana CV unit of `morae.MORAE`

Consonants are rule-based: closure silence + place-dependent burst + aspiration for stops, noise bands for fricatives,
damped nasal murmur with a zero, a tap for r, glides for y/w. Formant transitions follow vowel-dependent loci (locus
equations), so "ka" and "ki" start differently. Vowels are steady so the editor can stretch them. Every clip has a flat
.frq file at F0. The result is intentionally synthetic.

    python -m voice2utau.synthbank -o out --name MADO
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
F0 = 261.63                       # mean pitch of every sample (C4)
BLOCK = 64
VOWEL_MS = 480                    # steady vowel after the transition
# Japanese vowels, female speaker: (F1, F2, F3). u is the unrounded, central-ish [ɯ].
VOWELS = {"a": (850, 1400, 2900), "i": (330, 2750, 3350), "u": (340, 1150, 2500), "e": (560, 2300, 3000),
          "o": (620, 1000, 2700)}
F4, F5 = 3900, 4800
FORMANT_SCALE = 1.0               # vocal-tract size: every resonance (vowels, loci, nasals, breath) x this; ~0.86 = male
BW = np.array([80.0, 100.0, 150.0, 220.0, 300.0])


def locus(place: str, v: str) -> tuple[float, float, float]:
    """Formant values at consonant release (locus equations: F2onset = k * F2vowel + c)."""
    F1, F2, F3 = VOWELS[v]
    if place == "lab":
        return 260.0, 0.75 * F2 + 220, min(F3, 2300.0)
    if place == "alv":
        return 280.0, 0.45 * F2 + 1050, 2750.0
    if place == "vel":                                    # velar pinch: F2 and F3 close together
        f2 = 0.8 * F2 + 450 if F2 < 1800 else min(F2 + 200, 2900.0)
        return 280.0, f2, f2 + 300
    if place == "pal":
        return 280.0, 2450.0, 3150.0
    return float(F1), float(F2), float(F3)


# class -> (kind, place, closure ms, consonant ms after the closure (to voice/vowel onset), transition ms, extras)
CLASSES: dict[str, tuple] = {
    "": ("vowel", "", 0, 0, 0, {}),
    "k": ("stop_v", "vel", 45, 45, 45, {"vot": 40}),
    "ky": ("stop_v", "pal", 45, 52, 50, {"vot": 46}),
    "t": ("stop_v", "alv", 40, 36, 40, {"vot": 30}),
    "p": ("stop_v", "lab", 40, 30, 40, {"vot": 24}),
    "py": ("stop_v", "pal", 40, 38, 50, {"vot": 30, "bp": "lab"}),
    "g": ("stop_d", "vel", 40, 10, 45, {}),
    "gy": ("stop_d", "pal", 40, 14, 50, {}),
    "d": ("stop_d", "alv", 35, 8, 40, {}),
    "b": ("stop_d", "lab", 35, 8, 40, {}),
    "by": ("stop_d", "pal", 35, 12, 50, {"bp": "lab"}),
    "s": ("fric", "alv", 0, 110, 40, {"cf": 7300, "bw": 2800, "hp": 4500, "amp": 0.5, "db": -12}),
    "sh": ("fric", "pal", 0, 120, 60, {"cf": 3500, "bw": 2100, "hp": 2200, "amp": 0.75, "db": -12}),
    "z": ("fric_v", "alv", 0, 75, 40, {"cf": 7000, "bw": 2800, "hp": 4500, "amp": 0.3, "db": -15}),
    "j": ("aff_v", "pal", 35, 55, 60, {"cf": 3400, "bw": 2000, "hp": 2200, "amp": 0.4, "db": -14}),
    "ch": ("aff", "pal", 40, 62, 60, {"cf": 3500, "bw": 2100, "hp": 2200, "amp": 0.7, "db": -11}),
    "ts": ("aff", "alv", 40, 62, 40, {"cf": 7200, "bw": 2800, "hp": 4500, "amp": 0.45, "db": -11}),
    "h": ("h", "", 0, 95, 30, {}),
    "hy": ("hy", "pal", 0, 95, 50, {"cf": 2700, "bw": 1300, "hp": 1600, "amp": 0.12, "db": -15}),
    "f": ("f", "lab", 0, 90, 40, {"db": -18}),
    "n": ("nasal", "alv", 0, 90, 30, {"pole": 1500, "zero": 1700}),
    "ny": ("nasal", "pal", 0, 90, 50, {"pole": 2200, "zero": 2500}),
    "m": ("nasal", "lab", 0, 90, 30, {"pole": 1100, "zero": 1000}),
    "my": ("nasal", "pal", 0, 90, 50, {"pole": 1100, "zero": 1000}),
    "r": ("tap", "alv", 0, 30, 30, {}),
    "ry": ("tap", "pal", 0, 40, 50, {}),
    "y": ("glide", "y", 0, 45, 80, {}),
    "w": ("glide", "w", 0, 55, 90, {}),
}
GLIDE_START = {"y": (280.0, 2500.0, 3200.0), "w": (300.0, 720.0, 2300.0)}


def _ramp(n: int, a: int, b: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Piecewise-linear ramp from lo (before a) to hi (after b)."""
    return np.interp(np.arange(n), [a, max(b, a + 1)], [lo, hi])


def _track(loc: float, tgt: float, n: int, c: int, tr: int) -> np.ndarray:
    return np.interp(np.arange(n), [0, c, c + max(tr, 1), n], [loc, loc, tgt, tgt])


# Optional replacement sources (see voice2utau.ninchan): SOURCE_FN(n, rng) -> voiced excitation at F0,
# NOISE_FN(n, rng) -> broadband noise. None = the built-in glottal pulses / white noise.
SOURCE_FN = None
NOISE_FN = None
# BURST_FN(n_samples, place, rng) -> a stop/affricate release burst (e.g. a slice of a real button click)
BURST_FN = None
# POST_FN(y, vowel_onset_sample, mora, rng) -> (y, vowel_onset_sample): character touches added to each finished clip
# (it may prepend material, e.g. a whoosh, by returning a later vowel onset)
POST_FN = None


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return NOISE_FN(n, rng) if NOISE_FN is not None else rng.standard_normal(n)


def _source(n: int, rng: np.random.Generator, jitter: float = 0.004, shimmer: float = 0.03) -> np.ndarray:
    if SOURCE_FN is not None:
        return SOURCE_FN(n, rng)
    """Differentiated Rosenberg pulses, one period at a time: mean pitch F0, slight jitter/shimmer, gentle tilt."""
    out = np.zeros(n)
    i = 0
    while i < n:
        T = max(8, int(round(SR / (F0 * (1 + jitter * rng.standard_normal())))))
        tp, te = int(0.48 * T), int(0.6 * T)
        p = np.zeros(T)
        p[:tp] = 0.5 * (1 - np.cos(np.pi * np.arange(tp) / tp))
        p[tp:te] = np.cos(0.5 * np.pi * np.arange(te - tp) / max(1, te - tp))
        d = np.diff(p, prepend=0.0) * (1 + shimmer * rng.standard_normal())
        m = min(T, n - i)
        out[i:i + m] = d[:m]
        i += T
    out = lfilter([0.6], [1, -0.4], out)                      # soften the buzz a little
    return out / max(1e-9, np.abs(out).max())


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
            c = -np.exp(-2 * np.pi * bw / SR)
            b = 2 * np.exp(-np.pi * bw / SR) * np.cos(2 * np.pi * float(f[mid]) * FORMANT_SCALE / SR)
            out[s:e], zi = lfilter([1 - b - c], [1.0, -b, -c], y[s:e], zi=zi)
        y = out
    return y


def _antiresonance(x: np.ndarray, fz: float, bw: float, gate: np.ndarray) -> np.ndarray:
    """Nasal zero (Klatt anti-resonator), mixed in only where gate > 0."""
    c = -np.exp(-2 * np.pi * bw / SR)
    b = 2 * np.exp(-np.pi * bw / SR) * np.cos(2 * np.pi * fz * FORMANT_SCALE / SR)
    a = 1 - b - c
    z = lfilter([1 / a, -b / a, -c / a], [1.0], x)
    return x * (1 - gate) + z * gate


def _band_noise(n: int, cf: float, bw: float, hp: float, rng: np.random.Generator) -> np.ndarray:
    x = _noise(n, rng)
    lo, hi = max(200.0, cf - bw), min(SR / 2 - 500, cf + bw)
    x = sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)
    x = sosfilt(butter(2, hp, "high", fs=SR, output="sos"), x)
    return x / max(1e-9, np.abs(x).max())


def _burst_spec(place: str, v: str) -> tuple[float, float, float]:
    """(centre Hz, half-width Hz, level) of the release burst."""
    if place == "vel":
        return locus("vel", v)[1] + 150, 380, 1.3              # compact (narrow), near the vowel's F2
    if place == "pal":
        return 2700, 700, 1.1
    if place == "alv":
        return 4800, 2200, 0.9
    return 1200, 1100, 0.55                                    # labial: weak, diffuse, low


def synth(m: Mora, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Render one CV mora. Returns (audio at 44.1 kHz, preutterance in seconds = vowel/voicing onset)."""
    rng = rng or np.random.default_rng(1)
    if m.key == "nn":
        return _syllabic_n(rng)
    kind, place, cl_ms, c_ms, tr_ms, ex = CLASSES[m.cls]
    cl = int(cl_ms / 1000 * SR)
    c = cl + int(c_ms / 1000 * SR)                             # vowel (voicing) onset
    tr = int(tr_ms / 1000 * SR)
    n = c + tr + int(VOWEL_MS / 1000 * SR)
    v = VOWELS[m.vowel]
    if kind == "glide":
        loc = GLIDE_START[place]
    elif kind == "h":
        loc = v
    else:
        loc = locus(place, m.vowel) if place else v
    # formants start moving at the release for stops (for voiceless ones most of the move happens during aspiration,
    # otherwise the voiced transition sounds like l/d), at the end of the noise for fricatives
    t_on = cl if kind in ("stop_d", "stop_v") else c
    f = [_track(loc[0], v[0], n, t_on, tr), _track(loc[1], v[1], n, t_on, tr), _track(loc[2], v[2], n, t_on, tr),
         np.full(n, F4), np.full(n, F5)]
    idx = np.arange(n)
    glot = _source(n, rng)
    voiced = _ramp(n, c, c + int(0.008 * SR))
    asp = np.zeros(n)
    fric = np.zeros(n)
    bwscale = None
    gate_zero = np.zeros(n)
    zero_f = 0.0
    lp_gate = None
    asp_hp = 0.0

    if kind == "vowel":
        voiced = _ramp(n, 0, int(0.025 * SR))
    elif kind in ("stop_v", "stop_d"):
        bcf, bbw, blev = _burst_spec(ex.get("bp", place), m.vowel)
        bl = int(0.015 * SR)
        burst = (BURST_FN(bl, ex.get("bp", place), rng) if BURST_FN is not None
                 else _band_noise(bl, bcf, bbw, 400.0, rng) * np.exp(-np.arange(bl) / (bl / 2.5)))
        if kind == "stop_v":
            fric[cl:cl + bl] += burst * 4.0 * blev
            if place in ("vel", "pal"):                        # velars often release with a second, smaller burst
                k2 = cl + int(0.018 * SR)
                fric[k2:k2 + bl] += burst * 1.6 * blev
            a0 = cl + int(0.004 * SR)
            asp = ((idx >= a0) & (idx < c + int(0.004 * SR))) * _ramp(n, a0, a0 + int(0.006 * SR)) * 3.2
            asp_hp = 1200.0
        else:
            bar = np.where(idx < cl, 0.12, 1.0)                # low voicing bar through the closure
            voiced = bar * _ramp(n, 0, int(0.01 * SR))
            f_bar = [np.where(idx < cl, 230.0, x) if k == 0 else x for k, x in enumerate(f)]
            f = f_bar
            fric[cl:cl + bl] += burst * 1.8 * blev
            bwscale = np.where(idx < cl, 4.0, 1.0)             # the bar is mostly F1
    elif kind in ("fric", "fric_v", "aff", "aff_v", "hy"):
        f0n = cl
        noise = _band_noise(n, ex["cf"], ex["bw"], ex["hp"], rng)
        rise = int((0.004 if kind.startswith("aff") else 0.03) * SR)
        env = _ramp(n, f0n, f0n + rise) * (1 - _ramp(n, c - int(0.012 * SR), c + int(0.006 * SR)))
        if kind.startswith("aff"):
            bcf, bbw, _ = _burst_spec(place, m.vowel)
            bl = int(0.010 * SR)
            b = (BURST_FN(bl, place, rng) if BURST_FN is not None
                 else _band_noise(bl, bcf, bbw, 400.0, rng) * np.exp(-np.arange(bl) / (bl / 2.5)))
            fric[cl:cl + bl] += b * 5.0
        if kind.endswith("_v"):
            voiced = np.where(idx < c, 0.5, 1.0) * _ramp(n, 0, int(0.012 * SR))
            if kind == "aff_v":
                voiced = voiced * np.where(idx < cl, 0.25, 1.0)
            noise = noise * (0.6 + 0.4 * np.abs(glot) / max(1e-9, np.abs(glot).max()))   # voicing-modulated noise
        fric += noise * env * ex["amp"]
        if kind == "hy":
            asp = env * 1.6
    elif kind == "h":
        a = int(0.025 * SR)
        asp = _ramp(n, 0, a) * (1 - _ramp(n, c - int(0.01 * SR), c + int(0.01 * SR))) * 0.65
        asp_hp = 400.0
        bwscale = np.where(idx < c, 2.5, 1.0)                  # breathy, wide F1-F3 during [h]
        voiced = _ramp(n, c - int(0.012 * SR), c + int(0.012 * SR))
    elif kind == "f":
        a = int(0.025 * SR)
        envf = _ramp(n, 0, a) * (1 - _ramp(n, c - int(0.012 * SR), c + int(0.006 * SR)))
        fric += _band_noise(n, 2400, 2200, 600, rng) * envf * 0.22
        asp = envf * 0.5
    elif kind == "nasal":
        nas = (260.0, float(ex["pole"]), 2700.0)
        f = [np.where(idx < c, nas[k], x) if k < 3 else x for k, x in enumerate(f)]
        voiced = np.where(idx < c, 0.65, 1.0) * _ramp(n, 0, int(0.012 * SR))
        bwscale = np.where(idx < c, 3.0, 1.0)                 # damped F2/F3 in the murmur
        gate_zero = 1 - _ramp(n, c - int(0.004 * SR), c + int(0.012 * SR))
        zero_f = float(ex["zero"])
    elif kind == "tap":
        t = idx / SR
        tc = c / SR - 0.012
        dip = 1.0 - 0.9 * np.exp(-(((t - tc) / 0.007) ** 2))
        voiced = _ramp(n, 0, int(0.012 * SR)) * dip
        f = [np.where(idx < c, x * (0.85 if k == 2 else 1.0), x) for k, x in enumerate(f)]   # lowered F3 before the tap
    elif kind == "glide":
        voiced = _ramp(n, 0, int(0.03 * SR)) * np.where(idx < c, 0.75, 1.0)

    noise = _noise(n, rng)
    if asp_hp:                                                 # aspiration after a stop: open glottis, energy in F2/F3
        noise = sosfilt(butter(2, asp_hp, "high", fs=SR, output="sos"), noise) * 2.0
    src = glot * voiced + noise * asp * 0.3 + _noise(n, rng) * voiced * 0.004
    if lp_gate is not None:
        low = lfilter([1 - 0.93], [1, -0.93], src) * 1.4
        src = src * (1 - lp_gate) + low * lp_gate
    y = _cascade(src, f, bwscale)
    if zero_f:
        y = _antiresonance(y, zero_f, 120.0, gate_zero)
    fric = fric * 1.6
    if "db" in ex and c > int(0.02 * SR):
        # set the consonant noise relative to the vowel that follows (a fixed level made し/ち/じ, before the quiet
        # vowel い, almost as loud as the vowel itself)
        win = slice(c + int(0.05 * SR), c + int(0.15 * SR))
        rv = float(np.sqrt((y[win] ** 2).mean()))
        rf = float(np.sqrt((fric[max(0, cl - int(0.002 * SR)):c] ** 2).mean()))
        if rf > 0:
            fric = fric * (10 ** (ex["db"] / 20) * rv / rf)
    y = y + fric
    if kind != "vowel":
        y = cap_consonant(y, c + tr // 2, c + tr)
    tail = int(0.04 * SR)
    y[-tail:] *= np.linspace(1, 0, tail)
    split = c if kind != "vowel" else int(0.02 * SR)
    if POST_FN is not None:
        y, split = POST_FN(y, split, m, rng)
    y = y / max(1e-9, np.abs(y).max()) * 0.8
    return y.astype("float32"), split / SR


CONSONANT_CAP_DB = -1.0           # no 10 ms stretch of a consonant may be louder than its vowel


def cap_consonant(y: np.ndarray, end: int, vowel_at: int, cap_db: float = CONSONANT_CAP_DB) -> np.ndarray:
    """Limit the consonant part (before `end`) so its short-term level never exceeds the steady vowel's level."""
    w = int(0.01 * SR)
    vow = y[vowel_at:vowel_at + int(0.2 * SR)]
    if len(vow) < 4 * w or end <= 0:
        return y
    nv = len(vow) // w
    target = float(np.median(np.sqrt((vow[:nv * w].reshape(nv, w) ** 2).mean(1)))) * 10 ** (cap_db / 20)
    hop = w // 2
    starts = np.arange(0, max(1, end - w + 1), hop)
    lvl = np.array([np.sqrt((y[s:s + w] ** 2).mean()) for s in starts])
    g = np.minimum(1.0, target / np.maximum(lvl, 1e-12))
    gain = np.interp(np.arange(end), starts + w // 2, g)
    k = int(0.004 * SR)                                           # smooth the gain a little (no zipper noise)
    gain = np.minimum(gain, np.convolve(gain, np.ones(k) / k, "same"))
    y = y.copy()
    y[:end] *= gain
    return y


def _syllabic_n(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    n = int(0.45 * SR)
    f = [np.full(n, 260.0), np.full(n, 1200.0), np.full(n, 2600.0), np.full(n, F4), np.full(n, F5)]
    voiced = _ramp(n, 0, int(0.02 * SR)) * 0.8
    y = _cascade(_source(n, rng) * voiced, f, np.full(n, 3.0))
    y = _antiresonance(y, 1600.0, 150.0, np.ones(n))
    tail = int(0.05 * SR)
    y[-tail:] *= np.linspace(1, 0, tail)
    return (y / max(1e-9, np.abs(y).max()) * 0.8).astype("float32"), 0.02


def breath(rng: np.random.Generator) -> np.ndarray:
    """An in-breath: soft noise through a neutral vowel tract."""
    n = int(0.5 * SR)
    env = np.sin(np.pi * np.arange(n) / n) ** 1.5
    f = [np.full(n, 500.0), np.full(n, 1500.0), np.full(n, 2500.0), np.full(n, F4), np.full(n, F5)]
    y = _cascade(_noise(n, rng) * env, f, np.full(n, 3.0))
    return (y / max(1e-9, np.abs(y).max()) * 0.3).astype("float32")


def build(out_dir: Path, name: str = "MADO") -> Path:
    rng = np.random.default_rng(7)
    clips: dict[str, Clip] = {}
    for key, m in MORAE.items():
        y, split = synth(m, rng)
        y = audio.normalise_level(y)
        clips[key] = Clip(m, y, split, F0, "synth", 1.0, "klatt-style source-filter")
    report = {"kind": "synthbank", "f0_hz": F0, "units": len(clips),
              "samples": [{"key": k, "source": "synth", "f0": F0} for k in clips],
              "method": "glottal pulses + aspiration through a 5-formant cascade (+ nasal zero); band-passed noise "
                        "for fricatives and bursts; vowel-dependent consonant loci"}
    root = write_bank(clips, out_dir, name, report)
    for k in clips:
        write_frq(root / f"{k}_wav.frq", len(clips[k].audio))
    # romaji aliases next to the hiragana ones, plus a breath sample
    oto = root / "oto.ini"
    lines = oto.read_bytes().decode("cp932").splitlines()
    extra = []
    for l in lines:
        wav, rest = l.split("=", 1)
        key = wav[:-4]
        romaji = "n" if key == "nn" else key
        extra.append(f"{wav}={romaji},{rest.split(',', 1)[1]}")
    br = breath(rng)
    audio.write_wav(root / "br.wav", br, SR)
    write_frq(root / "br_wav.frq", len(br))
    dur = len(br) / SR * 1000
    extra += [f"br.wav=息,0,{dur - 10:.1f},{-dur:.1f},20,5", f"br.wav=br,0,{dur - 10:.1f},{-dur:.1f},20,5"]
    oto.write_bytes(("\n".join(lines + extra) + "\n").encode("cp932"))
    (root / "readme.txt").write_text(
        f"{name}\nA Japanese CV voicebank synthesised from scratch (source-filter synthesis, mean pitch C4 = {F0:.2f} Hz), "
        "no recordings and no voice model.\nAliases: hiragana (か) and romaji (ka); ん is also 'n'; 息 / br is a breath.\n"
        "Vowels are steady so long notes stretch cleanly; the voice is intentionally synthetic.\n"
        "Generated by voice2utau.synthbank - free to use.\n", encoding="utf-8")
    return root


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--name", default="MADO")
    a = ap.parse_args()
    root = build(a.out, a.name)
    print(f"wrote {len(MORAE)} units (+ romaji aliases and a breath) to {root}")


if __name__ == "__main__":
    main()
