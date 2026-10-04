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


def build_candidates(phones: list[Phone], src: str) -> list[Candidate]:
    syms = [classify(p.sym) for p in phones]
    n = len(phones)
    out: list[Candidate] = []

    def gap(a: int, b: int) -> float:
        return phones[b].start - phones[a].end

    def lead(i: int, want: float) -> float:
        if i > 0 and gap(i - 1, i) < PAUSE_S:
            return max(0.0, min(want, gap(i - 1, i) / 2))
        return want

    def trail_after(j: int) -> tuple[float, float]:
        """(trail seconds, gap to next phone)."""
        if j + 1 < n and gap(j, j + 1) < PAUSE_S:
            g = max(0.0, gap(j, j + 1))
            return min(0.15, g * 0.6), g
        return 0.15, PAUSE_S

    i = 0
    while i < n:
        s = syms[i]
        if s.kind == "V":
            m = MORAE.get(s.value)
            if m:
                tr, g = trail_after(i)
                st = phones[i].start - lead(i, 0.06)
                out.append(Candidate(m, src, st, phones[i].end + tr,
                                     st + 0.03, phones[i].conf, g))
            i += 1
        elif s.kind in ("C", "G"):
            cls, palatal = (s.value, s.palatal) if s.kind == "C" else ("y", False)
            j = i + 1
            if s.kind == "C" and j < n and syms[j].kind == "G" and gap(i, j) < MAX_CV_GAP_S:
                palatal = True
                j += 1
            if j < n and syms[j].kind == "V" and gap(j - 1, j) < MAX_CV_GAP_S:
                m = lookup(cls, syms[j].value, palatal)
                if m:
                    tr, g = trail_after(j)
                    st = phones[i].start - lead(i, 0.06)
                    split = (phones[j - 1].end + phones[j].start) / 2
                    conf = float(np.mean([p.conf for p in phones[i:j + 1]]))
                    out.append(Candidate(m, src, st, phones[j].end + tr, split, conf, g))
                i = j + 1
            else:
                # nasal in coda position -> ん
                nasal = s.kind == "C" and cls in ("n", "m")
                after_vowel = i > 0 and syms[i - 1].kind == "V" and gap(i - 1, i) < MAX_CV_GAP_S
                if nasal and after_vowel:
                    tr, g = trail_after(i)
                    out.append(Candidate(MORAE["nn"], src, phones[i].start - 0.05,
                                         phones[i].end + min(0.08, tr), phones[i].start, phones[i].conf, g))
                i += 1
        elif s.kind == "N":
            tr, g = trail_after(i)
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
    return c.conf + dur + isolated


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
    f0 = audio.median_f0(seg[len(seg) // 3:], sr)
    return Clip(c.mora, seg, split, f0, "recorded", c.conf, c.src)


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


def select_best(cands: list[Candidate], wavs: dict[str, Path], min_conf: float = 0.45,
                top_k: int = 6, validator=None) -> dict[str, Clip]:
    if validator is not None:
        validator.annotate(cands)
    best: dict[str, Clip] = {}
    for key, cs in rank_candidates(cands, min_conf, validator).items():
        top: tuple[float, Clip] | None = None
        for rank, c in enumerate(cs[:top_k]):
            clip = cut_clip(c, wavs[c.src])
            if clip is None:
                continue
            ok = acoustic_ok(clip)
            if ok <= 0:
                continue
            total = (validator.rank_score(c) if validator is not None else c.score) * ok
            if top is None or total > top[0]:
                top = (total, clip)
        if top:
            best[key] = top[1]
    return best
