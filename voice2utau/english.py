"""English (ARPAbet CVVC) voicebank profile.

Units
  CV  "k ae"   consonant -> vowel      file cv_k_ae.wav
  VC  "ae t"   vowel -> consonant      file vc_ae_t.wav
  V   "- ae"   vowel at phrase start   file v_ae.wav
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

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


# German recordings: only sounds with a close English counterpart are accepted. German /ʁ/, /x/, /ç/,
# /ts/, /pf/, /y/, /ʏ/, /ø/, /œ/ and the pure (monophthong) tense /e:/ /o:/ are deliberately dropped:
# put under an English label they would be audibly wrong.
_DE_VOWEL = {"ɪ": "ih", "i": "iy", "ʊ": "uh", "u": "uw", "ɛ": "eh", "ɔ": "ao", "a": "aa", "ä": "aa",
             "ɑ": "aa", "ə": "ah", "ɐ": "ah"}
_DE_DIPH = {"aɪ": "ay", "ai": "ay", "aʊ": "aw", "au": "aw", "ɔʏ": "oy", "ɔɪ": "oy"}
_DE_CONS = {"p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "ɡ": "g", "g": "g", "f": "f", "v": "v",
            "s": "s", "z": "z", "ʃ": "sh", "ʒ": "zh", "h": "hh", "m": "m", "n": "n", "ŋ": "ng", "l": "l",
            "j": "y", "tʃ": "ch", "dʒ": "jh"}
DE_PENALTY = 0.10      # English takes win unless the German one is clearly better


def classify_de(sym: str) -> Sym:
    """IPA token from a German recording -> English unit sound, or Sym('X') if there is no close match."""
    if sym in _DE_DIPH:
        return Sym("V", _DE_DIPH[sym])
    if sym in _DE_CONS:
        return Sym("C", _DE_CONS[sym])
    base = _STRIP.sub("", sym.replace("ʲ", ""))
    if base in _DE_DIPH:
        return Sym("V", _DE_DIPH[base])
    if base in _DE_CONS:
        return Sym("C", _DE_CONS[base])
    if base in _DE_VOWEL:
        return Sym("V", _DE_VOWEL[base])
    return Sym("X")


# Recording whose language per line is unknown (English and German interleaved): accept only tokens that
# keep the same meaning in both languages. Dropped: sounds that are German-only or whose English reading
# would be wrong for a German speaker (German /a/ is not English /æ/; German r is /ʁ/ or a trill; tense
# /e o/ are pure vowels; ü ö are not /uw eh/).
_MIXED_DROP = {"ʁ", "x", "ç", "ts", "pf", "ʏ", "ø", "œ", "y", "ɐ", "e", "o", "r", "ɾ", "a", "ä",
               "eː", "oː", "aː", "yː", "øː", "ɛː", "ɑ̃", "ɔ̃", "ɛ̃", "œ̃"}


def classify_mixed(sym: str) -> Sym:
    base = _STRIP.sub("", sym.replace("ʲ", ""))
    if sym in _MIXED_DROP or base in _MIXED_DROP:
        return Sym("X")
    return classify(sym)


CLASSIFIERS = {"en": classify, "de": classify_de, "mixed": classify_mixed}


# --------------------------------------------------------------------- candidates

def build_candidates(phones: list[Phone], src: str, source_lang: str = "en") -> list[Candidate]:
    classifier = CLASSIFIERS[source_lang]
    syms = [classifier(p.sym) for p in phones]
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
        c.penalty = DE_PENALTY if source_lang == "de" else 0.0
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


@lru_cache(maxsize=1)
def unit_weights() -> dict[str, float]:
    """Share of everyday English speech each unit accounts for (sums to 1). See tools/build_unit_freq.py."""
    p = Path(__file__).parent / "data" / "en_unit_freq.json"
    return json.loads(p.read_text(encoding="utf-8"))["units"] if p.exists() else {}


# The recogniser's confidence is miscalibrated for two sounds: /ɚ/ is emitted with low probability (mean
# 0.26) and /oʊ/ never exceeds ~0.65, so the general 0.7 cut-off silently drops every `er` and `ow` unit
# (7.4 % + 3.6 % of everyday speech). Measured on 12 min of LJSpeech with transcripts: er at 0.5-0.7 is
# 90 % correct (vs 96 % above 0.7); ow at 0.6-0.65 is 94 % correct. Other vowels are clearly *worse* in
# that band (aa 0.48, eh 0.46, iy 0.43, uh 0.08), so only these two are relaxed.
CONF_OVERRIDES = {"er": 0.5, "ow": 0.6}


def min_conf_for(m: Mora, default: float) -> float:
    return min(default, CONF_OVERRIDES.get(m.vowel, default))
