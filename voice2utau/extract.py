"""Turn recognised phones into CV mora candidates and pick the best clip for each mora."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio
from .morae import MORAE, Mora, classify, lookup
from .phonemes import Phone

PAUSE_S = 0.25          # gap between phones beyond which we treat it as a pause
MAX_CV_GAP_S = 0.12     # max gap between a consonant and its vowel


@dataclass
class Candidate:
    mora: Mora
    src: str            # id of the source file
    start: float
    end: float
    split: float        # consonant -> vowel boundary (absolute seconds)
    conf: float
    next_gap: float     # silence/gap between this mora's vowel and the next phone
    score: float = 0.0
    penalty: float = 0.0   # subtracted from the score (e.g. non-native-language source)
    # filled in by validate.Validator.annotate
    feat: tuple | None = None
    vmargin: float | None = None
    cmargin: float | None = None
    consensus: float | None = None


@dataclass
class Clip:
    mora: Mora
    audio: np.ndarray            # 44.1 kHz mono
    split_s: float               # consonant length inside the clip
    f0: float | None
    source: str = "recorded"     # recorded | rvc | template
    conf: float = 1.0
    origin: str = ""
    meta: dict = field(default_factory=dict)


def gap(phones: list[Phone], a: int, b: int) -> float:
    return phones[b].start - phones[a].end


def lead(phones: list[Phone], i: int, want: float) -> float:
    """How far before phone i a clip may start without running into the previous phone."""
    if i > 0 and gap(phones, i - 1, i) < PAUSE_S:
        return max(0.0, min(want, gap(phones, i - 1, i) / 2))
    return want


def trail_after(phones: list[Phone], j: int, want: float = 0.15) -> tuple[float, float]:
    """(how far a clip may extend after phone j, gap to the next phone)."""
    if j + 1 < len(phones) and gap(phones, j, j + 1) < PAUSE_S:
        g = max(0.0, gap(phones, j, j + 1))
        return min(want, g * 0.6), g
    return want, PAUSE_S


def build_candidates(phones: list[Phone], src: str) -> list[Candidate]:
    syms = [classify(p.sym) for p in phones]
    n = len(phones)
    out: list[Candidate] = []

    def lead_(i, want):
        return lead(phones, i, want)

    def trail_after_(j):
        return trail_after(phones, j)

    def gap_(a, b):
        return gap(phones, a, b)

    i = 0
    while i < n:
        s = syms[i]
        if s.kind == "V":
            m = MORAE.get(s.value)
            if m:
                tr, g = trail_after_(i)
                st = phones[i].start - lead_(i, 0.06)
                out.append(Candidate(m, src, st, phones[i].end + tr,
                                     st + 0.03, phones[i].conf, g))
            i += 1
        elif s.kind in ("C", "G"):
            cls, palatal = (s.value, s.palatal) if s.kind == "C" else ("y", False)
            j = i + 1
            if s.kind == "C" and j < n and syms[j].kind == "G" and gap_(i, j) < MAX_CV_GAP_S:
                palatal = True
                j += 1
            if j < n and syms[j].kind == "V" and gap_(j - 1, j) < MAX_CV_GAP_S:
                m = lookup(cls, syms[j].value, palatal)
                if m:
                    tr, g = trail_after_(j)
                    st = phones[i].start - lead_(i, 0.06)
                    split = (phones[j - 1].end + phones[j].start) / 2
                    conf = float(np.mean([p.conf for p in phones[i:j + 1]]))
                    out.append(Candidate(m, src, st, phones[j].end + tr, split, conf, g))
                i = j + 1
            else:
                # nasal in coda position -> ん
                nasal = s.kind == "C" and cls in ("n", "m")
                after_vowel = i > 0 and syms[i - 1].kind == "V" and gap_(i - 1, i) < MAX_CV_GAP_S
                if nasal and after_vowel:
                    tr, g = trail_after_(i)
                    out.append(Candidate(MORAE["nn"], src, phones[i].start - 0.05,
                                         phones[i].end + min(0.08, tr), phones[i].start, phones[i].conf, g))
                i += 1
        elif s.kind == "N":
            tr, g = trail_after_(i)
            out.append(Candidate(MORAE["nn"], src, phones[i].start - 0.05,
                                 phones[i].end + min(0.08, tr), phones[i].start, phones[i].conf, g))
            i += 1
        else:
            i += 1
    for c in out:
        c.start = max(0.0, c.start)
        c.score = prescore(c)
    return out


def prescore(c: Candidate) -> float:
    d = c.end - c.start
    if d < 0.10 or d > 0.80:
        return -1.0
    dur = 0.25 if 0.16 <= d <= 0.45 else 0.10
    isolated = 0.10 if c.next_gap >= 0.03 else -0.05
    return c.conf + dur + isolated - c.penalty


def cut_clip(c: Candidate, wav44: Path) -> Clip | None:
    """Cut the candidate out of the 44.1 kHz source, trim silence, fade, and measure it."""
    pad = 0.05
    x, sr = audio.read_wav(wav44, max(0.0, c.start - pad), (c.end - c.start) + 2 * pad)
    off = max(0.0, c.start - pad)
    a, b = int((c.start - off) * sr), int((c.end - off) * sr)
    seg = x[a:b]
    if len(seg) < sr * 0.08:
        return None
    ta, tb = audio.trim_silence(seg, sr)
    seg = seg[ta:tb]
    split = max(0.0, (c.split - c.start) - ta / sr)
    seg = audio.fade(seg, sr)
    peak = float(np.abs(seg).max())
    if peak < 10 ** (-45 / 20):
        return None
    voiced_part = seg[: 2 * len(seg) // 3] if c.mora.kind == "VC" else seg[len(seg) // 3:]
    f0 = audio.median_f0(voiced_part, sr)
    return Clip(c.mora, seg, split, f0, "recorded", c.conf, c.src, {"f0_std": f0_std_semitones(voiced_part, sr)})


def f0_std_semitones(x: np.ndarray, sr: int) -> float | None:
    """How much the pitch moves inside a clip (semitones, std). Calm, steady takes make better bank samples."""
    try:
        _, f0 = audio.f0_track(x, sr)
    except Exception:
        return None
    v = f0[f0 > 0]
    return float(np.std(12 * np.log2(v / np.median(v)))) if len(v) >= 6 else None


def acoustic_ok(clip: Clip, sr: int = audio.SR_BANK) -> float:
    """0..1 plausibility: the vowel part must be voiced and clipping-free."""
    if clip.mora.key == "nn":
        return 1.0 if clip.f0 else 0.0
    d = len(clip.audio) / sr
    if not clip.f0:
        return 0.0
    if d < 0.12 or d > 0.8:
        return 0.3
    return 1.0


def rank_candidates(cands: list[Candidate], min_conf: float = 0.45, validator=None) -> dict[str, list[Candidate]]:
    """Per mora, the usable candidates ordered best-first."""
    by_key: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.score <= 0 or c.conf < min_conf:
            continue
        if validator is not None and not validator.accept(c):
            continue
        by_key.setdefault(c.mora.key, []).append(c)
    keyf = (lambda c: validator.rank_score(c)) if validator is not None else (lambda c: c.score)
    for cs in by_key.values():
        cs.sort(key=keyf, reverse=True)
    return by_key


PITCH_WEIGHT = 0.04       # score lost per semitone away from the speaker's median pitch (capped at 12)
STABILITY_WEIGHT = 0.03   # score lost per semitone of pitch movement inside the clip (capped at 6)


@dataclass
class Option:
    """A cut-out clip that could represent a unit, with its pre-selection score."""
    total: float
    clip: Clip
    foreign: bool = False    # recorded in a secondary language (fallback only)


def collect_options(cands: list[Candidate], wavs: dict[str, Path], min_conf: float = 0.45,
                    top_k: int = 6, validator=None) -> dict[str, list[Option]]:
    """Cut the best `top_k` candidates of every unit out of the sources."""
    if validator is not None:
        validator.annotate(cands)
    options: dict[str, list[Option]] = {}
    for key, cs in rank_candidates(cands, min_conf, validator).items():
        for c in cs[:top_k]:
            clip = cut_clip(c, wavs[c.src])
            if clip is None:
                continue
            ok = acoustic_ok(clip)
            if ok <= 0:
                continue
            total = (validator.rank_score(c) if validator is not None else c.score) * ok
            options.setdefault(key, []).append(Option(total, clip, c.penalty > 0))
    return options


def scorer(options: dict[str, list[Option]]):
    """Score function used to rank options (also used to decide which options a corpus keeps)."""
    f0s = [o.clip.f0 for opts in options.values() for o in opts if o.clip.f0]
    median = float(np.median(f0s)) if f0s else None

    def adjusted(o: Option) -> float:
        total = o.total
        if median and o.clip.f0:
            total -= PITCH_WEIGHT * min(12.0, abs(12 * np.log2(o.clip.f0 / median)))
        std = o.clip.meta.get("f0_std")
        if std is not None:
            total -= STABILITY_WEIGHT * min(6.0, std)
        return total
    return adjusted


def choose(options: dict[str, list[Option]]) -> dict[str, Clip]:
    """Pick one clip per unit.

    Prefers clips near the speaker's typical pitch (little PSOLA shifting when flattened) and with
    steady pitch inside the clip. A recording in a secondary language (e.g. German for an English bank)
    is only a fallback: it is used for a unit only when no native-language candidate survived.
    """
    adjusted = scorer(options)

    def pick(opts: list[Option]) -> Clip:
        native = [o for o in opts if not o.foreign]
        return max(native or opts, key=adjusted).clip

    return {key: pick(opts) for key, opts in options.items() if opts}


def select_best(cands: list[Candidate], wavs: dict[str, Path], min_conf: float = 0.45,
                top_k: int = 6, validator=None) -> dict[str, Clip]:
    return choose(collect_options(cands, wavs, min_conf, top_k, validator))
