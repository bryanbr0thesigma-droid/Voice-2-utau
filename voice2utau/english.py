"""English (ARPAbet CVVC) voicebank profile.

Units
  CV  "k ae"   consonant -> vowel      file cv_k_ae.wav
  VC  "ae t"   vowel -> consonant      file vc_ae_t.wav
  V   "- ae"   vowel at phrase start   file v_ae.wav
"""
from __future__ import annotations

import numpy as np

from .extract import Candidate, gap, lead, trail_after, prescore
from .morae import Mora, Sym, _STRIP
from .phonemes import Phone

VOWELS = ["aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"]
CONSONANTS = ["b", "ch", "d", "dh", "f", "g", "hh", "jh", "k", "l", "m", "n", "ng", "p", "r", "s",
              "sh", "t", "th", "v", "w", "y", "z", "zh"]
NO_CV = {"ng"}                    # English syllables never start with /ŋ/
NO_VC = {"hh", "w", "y"}          # ... and these never end one
VOICELESS = {"p", "t", "k", "f", "th", "s", "sh", "ch", "hh"}

MAX_CV_GAP_S = 0.12


def _build() -> dict[str, Mora]:
    out: dict[str, Mora] = {}
    for c in CONSONANTS:
        if c in NO_CV:
            continue
        for v in VOWELS:
            out[f"cv_{c}_{v}"] = Mora(f"cv_{c}_{v}", f"{c} {v}", c, v, "CV", "en")
    for v in VOWELS:
        for c in CONSONANTS:
            if c in NO_VC:
                continue
            out[f"vc_{v}_{c}"] = Mora(f"vc_{v}_{c}", f"{v} {c}", c, v, "VC", "en")
    for v in VOWELS:
        out[f"v_{v}"] = Mora(f"v_{v}", f"- {v}", "", v, "V", "en")
    return out


UNITS: dict[str, Mora] = _build()


def group_of(m: Mora) -> str:
    """Row label used by the UI grid."""
    return {"CV": f"CV · {m.cls}", "VC": f"VC · {m.vowel}", "V": "vowels"}[m.kind]


# --------------------------------------------------------------------- IPA -> ARPAbet

_VOWEL_MAP = {
    "ɑ": "aa", "ɒ": "aa", "ä": "aa", "æ": "ae", "a": "ae", "ʌ": "ah", "ə": "ah", "ɐ": "ah",
    "ɔ": "ao", "o": "ow", "ɛ": "eh", "e": "eh", "ɜ": "er", "ɚ": "er", "ɝ": "er",
    "ɪ": "ih", "ᵻ": "ih", "ɨ": "ih", "i": "iy", "ʊ": "uh", "u": "uw", "ʉ": "uw", "ɯ": "uw", "y": "uw",
}
_DIPH = {"aɪ": "ay", "ai": "ay", "aʊ": "aw", "au": "aw", "eɪ": "ey", "ei": "ey", "oʊ": "ow", "əʊ": "ow",
         "ɔɪ": "oy", "oɪ": "oy", "ɑːɹ": "aa", "ɔːɹ": "ao", "oːɹ": "ao", "ɛɹ": "eh", "ɪɹ": "ih",
         "ʊɹ": "uh", "əl": "ah", "aɪɚ": "ay", "aɪə": "ay", "iə": "iy"}
_CONS_MAP = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "ɡ": "g", "g": "g", "ʔ": "",
    "tʃ": "ch", "dʒ": "jh", "f": "f", "v": "v", "θ": "th", "ð": "dh", "s": "s", "z": "z",
    "ʃ": "sh", "ʒ": "zh", "h": "hh", "m": "m", "n": "n", "ŋ": "ng", "l": "l", "ɫ": "l", "ɭ": "l",
    "ɹ": "r", "r": "r", "ɾ": "d", "ʁ": "r", "w": "w", "j": "y", "ɕ": "sh", "ɟ": "g", "c": "k", "x": "hh",
}


def classify(sym: str) -> Sym:
    """IPA token from the recogniser -> Sym(kind 'V'|'C'|'X', value=ARPAbet name in lower case)."""
    if sym in _DIPH:
        return Sym("V", _DIPH[sym])
    base = _STRIP.sub("", sym.replace("ʲ", ""))
    if base in _DIPH:
        return Sym("V", _DIPH[base])
    if base in _CONS_MAP:
        v = _CONS_MAP[base]
        return Sym("C", v) if v else Sym("X")
    if base in _VOWEL_MAP:
        return Sym("V", _VOWEL_MAP[base])
    if base and base[0] in _VOWEL_MAP and len(base) <= 3:      # other r-coloured / odd vowels
        return Sym("V", _VOWEL_MAP[base[0]])
    return Sym("X")


# --------------------------------------------------------------------- candidates

def build_candidates(phones: list[Phone], src: str) -> list[Candidate]:
    syms = [classify(p.sym) for p in phones]
    n = len(phones)
    out: list[Candidate] = []

    def adjacent(a: int, b: int) -> bool:
        return 0 <= a < n and 0 <= b < n and gap(phones, a, b) < MAX_CV_GAP_S

    for i in range(n):
        if syms[i].kind != "V":
            continue
        v = syms[i].value
        prev_c = i > 0 and syms[i - 1].kind == "C" and adjacent(i - 1, i)
        next_c = i + 1 < n and syms[i + 1].kind == "C" and adjacent(i, i + 1)
        vp = phones[i]

        if prev_c:                                          # CV: consonant onset into the vowel
            c = syms[i - 1].value
            key = f"cv_{c}_{v}"
            if key in UNITS:
                tr, g = trail_after(phones, i, 0.04)
                st = phones[i - 1].start - lead(phones, i - 1, 0.06)
                split = (phones[i - 1].end + vp.start) / 2
                conf = float(np.mean([phones[i - 1].conf, vp.conf]))
                out.append(Candidate(UNITS[key], src, st, vp.end + tr, split, conf, g))
        else:                                               # utterance-initial vowel
            key = f"v_{v}"
            tr, g = trail_after(phones, i, 0.10)
            st = vp.start - lead(phones, i, 0.06)
            out.append(Candidate(UNITS[key], src, st, vp.end + tr, st + 0.03, vp.conf, g))

        if next_c:                                          # VC: vowel into the following consonant
            c = syms[i + 1].value
            key = f"vc_{v}_{c}"
            if key in UNITS:
                cp = phones[i + 1]
                tr, g = trail_after(phones, i + 1, 0.08)
                st = vp.start - lead(phones, i, 0.06)
                split = (vp.end + cp.start) / 2
                conf = float(np.mean([vp.conf, cp.conf]))
                out.append(Candidate(UNITS[key], src, st, cp.end + tr, split, conf, g))
    for c in out:
        c.start = max(0.0, c.start)
        c.score = prescore(c)
    return out


# --------------------------------------------------------------------- espeak-ng synthesis

_ESPEAK_V = {"aa": "A:", "ae": "a", "ah": "V", "ao": "O:", "aw": "aU", "ay": "aI", "eh": "E", "er": "3:",
             "ey": "eI", "ih": "I", "iy": "i:", "ow": "oU", "oy": "OI", "uh": "U", "uw": "u:"}
_ESPEAK_C = {"b": "b", "ch": "tS", "d": "d", "dh": "D", "f": "f", "g": "g", "hh": "h", "jh": "dZ", "k": "k",
             "l": "l", "m": "m", "n": "n", "ng": "N", "p": "p", "r": "r", "s": "s", "sh": "S", "t": "t",
             "th": "T", "v": "v", "w": "w", "y": "j", "z": "z", "zh": "Z"}


def espeak_phonemes(m: Mora) -> str:
    """Espeak phoneme-mnemonic string ([[...]] input) that says this unit."""
    v = _ESPEAK_V[m.vowel]
    if m.kind == "CV":
        return f"{_ESPEAK_C[m.cls]}'{v}"
    if m.kind == "VC":
        return f"'{v}{_ESPEAK_C[m.cls]}"
    return f"'{v}"


# Fallback boundary estimates (ms) for clips we have no alignment for.
_PRE_MS = {"b": 45, "d": 45, "g": 50, "p": 60, "t": 60, "k": 70, "ch": 95, "jh": 80, "f": 90, "v": 60,
           "th": 80, "dh": 50, "s": 110, "z": 80, "sh": 115, "zh": 90, "hh": 70, "m": 55, "n": 50,
           "l": 50, "r": 55, "w": 55, "y": 55, "": 25}
_TAIL_MS = {"b": 80, "d": 80, "g": 80, "p": 100, "t": 100, "k": 110, "ch": 130, "jh": 120, "f": 120,
            "v": 90, "th": 110, "dh": 70, "s": 140, "z": 110, "sh": 150, "zh": 120, "m": 90, "n": 80,
            "ng": 90, "l": 80, "r": 80}


def default_split(m: Mora, dur_s: float) -> float:
    if m.kind == "VC":
        return max(0.06, dur_s - _TAIL_MS.get(m.cls, 90) / 1000)
    return min(_PRE_MS.get(m.cls, 60) / 1000, dur_s * 0.5)
