"""English (Arpasing) voicebanks from the same synthesiser as the CV banks: one recipe per character, every unit
synthesised, nothing recorded.

Arpasing is the ARPAbet English layout OpenUtau's Arpasing phonemizer reads: you type English words, it picks the
units. A bank needs, for 17 vowels and 24 consonants:
    "- V"  "V -"  "- C"  "C -"  "C V"  "V C"  "V V"  "C C"
Each unit is rendered here as a short phone sequence on one timeline (formants glide between the phones' targets,
voicing / aspiration / frication / bursts / nasal zeros follow the phones), at one reference loudness for the whole
bank so units join without level jumps. The character recipe (voice source, noise, click bursts, finishing touches)
plugs into the same hooks the CV banks use, so MADO, nin-Chan, BAKO, Kaso and ANDRO keep their sound in English.

    python -m voice2utau.english build CHARACTER OUT    (mado | ninchan | bako | kaso | andro)
"""
from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import butter, sosfilt

from . import audio, synthbank as sb

SR = sb.SR

# American English vowels, female reference (F1, F2, F3) after Hillenbrand et al. (1995); diphthongs glide to a
# second target. The synthesiser's FORMANT_SCALE shrinks or stretches everything for other vocal tracts.
VOWELS = {
    "iy": ((437, 2761, 3372), None), "ih": ((483, 2365, 3053), None), "ey": ((536, 2530, 3047), (440, 2700, 3300)),
    "eh": ((731, 2058, 2979), None), "ae": ((669, 2349, 2972), None), "aa": ((936, 1551, 2815), None),
    "ao": ((781, 1136, 2824), None), "ow": ((555, 1035, 2828), (470, 950, 2700)), "uh": ((519, 1225, 2827), None),
    "uw": ((459, 1105, 2735), None), "ah": ((753, 1426, 2933), None), "er": ((523, 1588, 1929), None),
    "ax": ((600, 1550, 2800), None), "ay": ((900, 1500, 2850), (480, 2400, 3000)),
    "aw": ((900, 1450, 2850), (500, 1050, 2700)), "oy": ((600, 950, 2800), (480, 2300, 3000)),
}
# kind, place, voiced, onset duration ms, extras
CONS = {
    "p": ("stop", "lab", False, 0, {"vot": 55}), "b": ("stop", "lab", True, 0, {}),
    "t": ("stop", "alv", False, 0, {"vot": 60}), "d": ("stop", "alv", True, 0, {}),
    "k": ("stop", "vel", False, 0, {"vot": 65}), "g": ("stop", "vel", True, 0, {}),
    "ch": ("aff", "pal", False, 75, {"cf": 3500, "bw": 2100, "hp": 2200, "db": -10}),
    "jh": ("aff", "pal", True, 70, {"cf": 3400, "bw": 2000, "hp": 2200, "db": -11}),
    "f": ("fric", "lab", False, 105, {"cf": 4500, "bw": 3800, "hp": 1200, "db": -20}),
    "v": ("fric", "lab", True, 75, {"cf": 4000, "bw": 3500, "hp": 1200, "db": -28, "vb": 0.85}),
    "th": ("fric", "den", False, 100, {"cf": 5000, "bw": 4000, "hp": 1500, "db": -27}),
    "dh": ("fric", "den", True, 50, {"cf": 4500, "bw": 3800, "hp": 1500, "db": -25, "vb": 0.75}),
    "s": ("fric", "alv", False, 115, {"cf": 7300, "bw": 2800, "hp": 4500, "db": -12}),
    "z": ("fric", "alv", True, 90, {"cf": 7000, "bw": 2800, "hp": 4500, "db": -18, "vb": 0.85}),
    "sh": ("fric", "pal", False, 120, {"cf": 3500, "bw": 2100, "hp": 2200, "db": -11}),
    "zh": ("fric", "pal", True, 90, {"cf": 3400, "bw": 2000, "hp": 2200, "db": -16, "vb": 0.85}),
    "hh": ("h", "", False, 85, {}),
    "m": ("nasal", "lab", True, 95, {"pole": 1100, "zero": 900}),
    "n": ("nasal", "alv", True, 95, {"pole": 1600, "zero": 1800}),
    "ng": ("nasal", "vel", True, 95, {"pole": 2200, "zero": 3000}),
    "l": ("liquid", "alv", True, 70, {"f": (380, 1050, 2900)}),
    "r": ("liquid", "alv", True, 90, {"f": (400, 1150, 1500)}),
    "w": ("glide", "lab", True, 60, {"f": (320, 760, 2300)}),
    "y": ("glide", "pal", True, 55, {"f": (300, 2550, 3250)}),
}
VOWEL_LIST = list(VOWELS)
CONS_LIST = list(CONS)
F4, F5 = 3900.0, 4800.0
TR = int(0.045 * SR)                  # formant transition length at a boundary
CLOSURE = int(0.055 * SR)


def is_vowel(p: str) -> bool:
    return p in VOWELS


def locus(place: str, v: str) -> tuple[float, float, float]:
    """Formants at a consonant boundary next to vowel v (locus equations, English values)."""
    (F1, F2, F3), _ = VOWELS[v]
    if place == "lab":
        return 300.0, 0.6 * F2 + 400, min(F3, 2500.0)
    if place == "den":
        return 320.0, 0.5 * F2 + 900, 2800.0
    if place == "alv":
        return 320.0, 0.45 * F2 + 1100, 2900.0
    if place == "vel":                                # the velar pinch: F2 and F3 close together, high before back vowels
        f2 = max(0.6 * F2 + 900, 1550.0) if F2 < 1800 else min(F2 + 250, 3000.0)
        return 320.0, f2, f2 + 300
    if place == "pal":
        return 320.0, 2300.0, 3000.0
    return float(F1), float(F2), float(F3)


@dataclasses.dataclass
class Seg:
    p: str
    s: int
    e: int
    sustain: bool = False             # a vowel held steady to the end (the stretchable part of the unit)


def plan(phones: list[str], sustain_ms: float = 450.0, lead_ms: float = 170.0, sil_ms: float = 45.0) -> list[Seg]:
    """Lay the phones of one unit on a timeline. The last phone, if a vowel, is a long steady sustain."""
    segs, t = [], 0
    for i, p in enumerate(phones):
        if p == "-":
            d = int(sil_ms / 1000 * SR) if i == 0 else int(0.12 * SR)
        elif is_vowel(p):
            last = i == len(phones) - 1
            d = int((sustain_ms if last else lead_ms) / 1000 * SR)
        else:
            kind = CONS[p][0]
            nxt = phones[i + 1] if i + 1 < len(phones) else "-"
            if kind == "stop":
                d = CLOSURE + (int(CONS[p][4].get("vot", 12) / 1000 * SR) if nxt != "-" or i > 0 else 0)
                if nxt == "-" or (i + 1 < len(phones) and not is_vowel(nxt)):
                    d = CLOSURE + int(0.06 * SR)              # released into a cluster or a pause
            elif kind == "aff":
                d = int(0.04 * SR) + int(CONS[p][3] / 1000 * SR)
            else:
                d = int(CONS[p][3] / 1000 * SR)
        segs.append(Seg(p, t, t + d, sustain=is_vowel(p) and i == len(phones) - 1))
        t += d
    return segs


def _neighbour_vowel(segs: list[Seg], i: int) -> str:
    for j in list(range(i + 1, len(segs))) + list(range(i - 1, -1, -1)):
        if is_vowel(segs[j].p):
            return segs[j].p
    return "ah"


def render(phones: list[str], rng: np.random.Generator, ref_rms: float | None = None) -> tuple[np.ndarray, list[Seg]]:
    """Synthesise one phone sequence ('-' = silence). Returns (audio, segments). With ref_rms, frication and bursts are
    set relative to that vowel loudness and the whole clip is scaled so a vowel lands at it."""
    segs = plan(phones)
    n = segs[-1].e
    idx = np.arange(n)
    # formant breakpoints, per formant: hold each phone's target inside it, glide across boundaries. After a nasal F1
    # jumps (F2/F3 still glide, which carries the place); after p/t/k the glide happens during the aspiration; after
    # r / w / y the vowel takes its time to arrive.
    pts: list[list[tuple[int, float]]] = [[], [], []]

    def put(at: int, tgt, which=(0, 1, 2)):
        for k in which:
            pts[k].append((int(at), float(tgt[k])))

    for i, g in enumerate(segs):
        if g.p == "-":
            continue
        prev = segs[i - 1].p if i > 0 else "-"
        if is_vowel(g.p):
            a, b = VOWELS[g.p]
            pk = CONS[prev][0] if prev in CONS else ""
            start = g.s + TR // 2
            if prev in ("r", "w", "y", "l"):
                start = g.s + int(0.07 * SR)
            if pk == "nasal":
                put(g.s + int(0.008 * SR), a, (0,))
                put(start, a, (1, 2))
            elif pk == "stop" and not CONS[prev][2]:
                put(g.s + int(0.01 * SR), a)                 # transition already done during the aspiration
            else:
                put(start, a)
            put(max(start + 1, g.e - TR // 2), b if (b and not g.sustain) else a)
            continue
        kind, place, voiced, _, ex = CONS[g.p]
        v = _neighbour_vowel(segs, i)
        if kind in ("liquid", "glide"):
            tgt = ex["f"]
        elif kind == "nasal":                         # murmur at the place's locus: the F2 glide into the vowel tells m/n/ng apart
            lf = locus(place, v)
            tgt = (250.0, lf[1], lf[2])
        elif kind == "h":
            tgt = VOWELS[v][0]
        else:
            tgt = locus(place, v)
        mid = (g.s + g.e) // 2
        hold = TR // 6 if g.p == "r" else TR // 3
        end = max(mid, g.e - hold)
        if kind == "stop" and not voiced:
            end = min(end, g.s + CLOSURE + int(0.005 * SR))   # glide to the vowel through the aspiration
        put(min(mid, g.s + hold), tgt)
        put(end, tgt)
    if not pts[0]:
        put(0, VOWELS["ah"][0])
    f = []
    for k in range(3):
        q = sorted(pts[k])
        f.append(np.interp(idx, [x for x, _ in q], [y for _, y in q]))
    f += [np.full(n, F4), np.full(n, F5)]

    voiced = np.zeros(n)
    asp = np.zeros(n)
    asp_hp = np.zeros(n)
    fric = np.zeros(n)
    bws = np.ones(n)
    zeros: list[tuple[float, np.ndarray]] = []
    glot = sb._source(n, rng)
    ref = ref_rms or 0.1

    def ramp_mask(s: int, e: int, r: int = int(0.006 * SR)) -> np.ndarray:
        m = np.zeros(n)
        m[s:e] = 1.0
        return np.convolve(m, np.ones(r) / r, "same") if r > 1 else m

    for i, g in enumerate(segs):
        if g.p == "-":
            continue
        if is_vowel(g.p):
            voiced[g.s:g.e] = 1.0
            continue
        kind, place, vd, _, ex = CONS[g.p]
        v = _neighbour_vowel(segs, i)
        nxt = segs[i + 1].p if i + 1 < len(segs) else "-"
        if kind == "stop":
            rel = g.s + CLOSURE
            if vd:
                voiced[g.s:rel] = 0.12
                f[0][g.s:rel] = 230.0
                bws[g.s:rel] = 4.0
                voiced[rel:g.e] = 1.0 if is_vowel(nxt) else 0.0
            released = rel < g.e
            if released:
                bl = int(0.015 * SR)
                cf, bw, lev = sb._burst_spec(place if place != "den" else "alv", "a")
                if place == "vel":
                    cf = locus("vel", v)[1] + 150
                burst = (sb.BURST_FN(bl, place, rng) if sb.BURST_FN is not None
                         else sb._band_noise(bl, cf, bw, 400.0, rng) * np.exp(-np.arange(bl) / (bl / 2.5)))
                k = min(bl, n - rel)
                fric[rel:rel + k] += burst[:k] * ref * (5.5 if not vd else 3.0) * lev * (1.6 if place == "vel" else 1.0)
                if place == "vel":
                    k2 = rel + int(0.016 * SR)
                    m2 = min(bl, n - k2)
                    if m2 > 0:
                        fric[k2:k2 + m2] += burst[:m2] * ref * 2.2 * lev
                if not vd:
                    a0 = rel + int(0.004 * SR)
                    asp += ramp_mask(a0, g.e) * 3.0
                    asp_hp[a0:g.e] = 1200.0
        elif kind in ("fric", "aff"):
            s0 = g.s
            if kind == "aff":
                s0 = g.s + int(0.04 * SR)                     # closure, then a burst into the fricative
                bl = int(0.010 * SR)
                b = (sb.BURST_FN(bl, place, rng) if sb.BURST_FN is not None
                     else sb._band_noise(bl, 3000, 900, 400.0, rng) * np.exp(-np.arange(bl) / (bl / 2.5)))
                k = min(bl, n - s0)
                fric[s0:s0 + k] += b[:k] * ref * 4.0
                if vd:
                    voiced[g.s:s0] = 0.12
            noise = sb._band_noise(n, ex["cf"], ex["bw"], ex["hp"], rng)
            env = ramp_mask(s0, g.e, int((0.004 if kind == "aff" else 0.02) * SR))
            if vd:
                voiced[s0:g.e] = np.maximum(voiced[s0:g.e], ex.get("vb", 0.6))
                noise = noise * (0.6 + 0.4 * np.abs(glot) / max(1e-9, np.abs(glot).max()))
            nr = float(np.sqrt((noise[s0:g.e] ** 2).mean())) if g.e > s0 else 1.0
            fric += noise * env * (ref * 10 ** (ex["db"] / 20) / max(1e-9, nr))
        elif kind == "h":
            asp += ramp_mask(g.s, g.e, int(0.015 * SR)) * 1.7
            fr = sb._band_noise(n, 3000, 2500, 1000, rng)            # a little glottal friction on top
            fric += fr * ramp_mask(g.s, g.e, int(0.015 * SR)) * ref * 10 ** (-22 / 20) / max(1e-9, float(np.sqrt((fr ** 2).mean())))
            asp_hp[g.s:g.e] = 400.0
            bws[g.s:g.e] = 2.5
        elif kind == "nasal":
            voiced[g.s:g.e] = 0.8
            bws[g.s:g.e] = 2.5
            zeros.append((float(ex["zero"]), ramp_mask(g.s, g.e, int(0.01 * SR))))
        elif kind == "liquid":
            voiced[g.s:g.e] = 0.85 if g.p == "l" else 1.0
            bws[g.s:g.e] = 1.6 if g.p == "l" else 0.8
        elif kind == "glide":
            voiced[g.s:g.e] = 0.85
    voiced = np.convolve(voiced, np.ones(int(0.006 * SR)) / int(0.006 * SR), "same")
    noise = sb._noise(n, rng)
    nz = np.zeros(n)
    for hp in (400.0, 1200.0):
        m = asp_hp == hp
        if m.any():
            nz[m] = sosfilt(butter(2, hp, "high", fs=SR, output="sos"), noise)[m] * 2.0
    src = glot * voiced + nz * asp * 0.3 + sb._noise(n, rng) * voiced * 0.004
    y = sb._cascade(src, f, bws)
    for fz, gate in zeros:
        y = sb._antiresonance(y, fz, 120.0, gate)
    if ref_rms:
        y = y * (ref_rms / VOICED_RMS[0]) if VOICED_RMS[0] else y
    y = y + fric
    return y, segs


VOICED_RMS = [0.0]                    # raw RMS of a steady reference vowel for the current recipe (set by calibrate)


def calibrate(rng: np.random.Generator, target: float = 0.12) -> float:
    """Measure the current recipe's raw vowel loudness so every unit can be scaled to the same vowel level."""
    VOICED_RMS[0] = 0.0
    y, segs = render(["aa"], rng)
    g = segs[0]
    VOICED_RMS[0] = float(np.sqrt((y[g.s + int(0.1 * SR):g.e - int(0.05 * SR)] ** 2).mean()))
    return target


# ---------------------------------------------------------------------------------------------- units and the bank
def units() -> list[list[str]]:
    V, C = VOWEL_LIST, CONS_LIST
    out = [["-", v] for v in V] + [[v, "-"] for v in V] + [["-", c] for c in C] + [[c, "-"] for c in C]
    out += [[c, v] for c in C for v in V] + [[v, c] for v in V for c in C] + [[a, b] for a in V for b in V]
    out += [[a, b] for a in C for b in C if a != b]
    return out


def alias(u: list[str]) -> str:
    return " ".join(u)


def fname(u: list[str]) -> str:
    return "_".join("sil" if p == "-" else p for p in u)


@dataclasses.dataclass
class Recipe:
    name: str
    f0: float = sb.F0
    scale: float = 1.0
    source: Callable | None = None
    noise: Callable | None = None
    burst: Callable | None = None
    onset_post: Callable | None = None   # (y, vowel_onset, rng) -> (y, vowel_onset): units that start a syllable
    all_post: Callable | None = None     # (y) -> y: every unit
    readme: str = ""


@contextlib.contextmanager
def using(r: Recipe):
    old = (sb.F0, sb.FORMANT_SCALE, sb.SOURCE_FN, sb.NOISE_FN, sb.BURST_FN, sb.POST_FN)
    sb.F0, sb.FORMANT_SCALE = r.f0, r.scale
    sb.SOURCE_FN, sb.NOISE_FN, sb.BURST_FN, sb.POST_FN = r.source, r.noise, r.burst, None
    try:
        yield
    finally:
        sb.F0, sb.FORMANT_SCALE, sb.SOURCE_FN, sb.NOISE_FN, sb.BURST_FN, sb.POST_FN = old


def oto_fields(u: list[str], segs: list[Seg], n: int, shift: int = 0) -> tuple[float, float, float, float]:
    """(consonant, cutoff, preutterance, overlap) in ms for one unit."""
    ms = lambda x: x / SR * 1000
    dur = ms(n)
    a, b = u
    second = segs[1].s + shift                               # where the unit's second phone starts
    if b == "-":                                             # an ending: aligned where the release starts
        pre = ms(second)
        return min(dur - 5, pre + 60), -dur, pre, min(40.0, pre * 0.5)
    if a == "-":                                             # a start from silence
        pre = ms(second)
        return min(dur - 5, pre + (60 if is_vowel(b) else 40)), -dur, pre, 0.0
    pre = ms(second)
    ovl = min(pre * 0.4, 45.0) if is_vowel(a) else min(pre * 0.35, 30.0)
    return min(dur - 5, pre + (60 if is_vowel(b) else 35)), -dur, pre, ovl


def build(recipe: Recipe, out: Path, rng_seed: int = 11, only: list[list[str]] | None = None) -> Path:
    from .sfxbank import write_frq
    root = Path(out) / recipe.name
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    lines = []
    with using(recipe):
        ref = calibrate(rng)
        for u in (only or units()):
            y, segs = render(u, rng, ref)
            shift = 0
            if recipe.onset_post and (u[0] == "-" or not is_vowel(u[0]) or is_vowel(u[1])) and is_vowel(u[-1]):
                vo = segs[-1].s
                w = slice(vo + int(0.06 * SR), vo + int(0.25 * SR))
                before = float(np.sqrt((y[w] ** 2).mean()))
                y2, vo2 = recipe.onset_post(y, vo, rng)
                shift = vo2 - vo
                w2 = slice(vo2 + int(0.06 * SR), vo2 + int(0.25 * SR))
                after = float(np.sqrt((y2[w2] ** 2).mean()))
                y = y2 * (before / after if after > 0 else 1.0)       # touches may re-normalise: keep the bank's level
            if recipe.all_post:
                y = recipe.all_post(y)
            tail = int(0.03 * SR)
            y = y.copy()
            y[-tail:] *= np.linspace(1, 0, tail)
            peak = float(np.abs(y).max())
            if peak > 0.98:
                y = y * 0.98 / peak
            nm = fname(u)
            audio.write_wav(root / f"{nm}.wav", y.astype("float32"), SR)
            write_frq(root / f"{nm}_wav.frq", len(y), hz=recipe.f0)
            con, cut, pre, ovl = oto_fields(u, segs, len(y), shift)
            lines.append(f"{nm}.wav={alias(u)},0,{con:.1f},{cut:.1f},{pre:.1f},{ovl:.1f}")
    (root / "oto.ini").write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932"))
    (root / "character.txt").write_bytes(f"name={recipe.name}\r\nimage=\r\nauthor=\r\nweb=\r\n".encode("cp932"))
    (root / "character.yaml").write_text(
        f"text_file_encoding: shift_jis\nportrait_opacity: 0.67\ndefault_phonemizer: OpenUtau.Plugin.Builtin.ArpasingPhonemizer\n",
        encoding="utf-8")
    (root / "readme.txt").write_text(
        f"{recipe.name}\nAn English (Arpasing) voicebank: {len(lines)} units ('- V', 'V -', '- C', 'C -', 'C V', 'V C', "
        f"'V V', 'C C') for {len(VOWEL_LIST)} vowels and {len(CONS_LIST)} consonants, all synthesised.\n"
        "Use it in OpenUtau with the Arpasing phonemizer (set as this bank's default): type English words as lyrics.\n"
        f"{recipe.readme}\nPitch: samples at {recipe.f0:.2f} Hz. Same synthesiser as the Japanese bank.\n",
        encoding="utf-8")
    return root


def main() -> None:
    import argparse
    from . import english_recipes
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("character", choices=sorted(english_recipes.RECIPES))
    b.add_argument("out", type=Path)
    a = ap.parse_args()
    root = build(english_recipes.RECIPES[a.character](), a.out)
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
