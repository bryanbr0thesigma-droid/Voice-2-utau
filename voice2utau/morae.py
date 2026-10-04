"""Target mora inventory (Japanese CV voicebank) and IPA -> mora mapping."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Mora:
    key: str      # romaji, also the wav file name stem
    kana: str     # alias written to oto.ini (hiragana for Japanese, ARPAbet pair for English)
    cls: str      # consonant class ('' for plain vowels)
    vowel: str    # a i u e o ('' for ん)
    kind: str = "CV"    # CV | VC | V   (VC only exists in the English bank)
    lang: str = "ja"


# (consonant class, [(vowel, kana), ...])
_ROWS: list[tuple[str, list[tuple[str, str]]]] = [
    ("", [("a", "あ"), ("i", "い"), ("u", "う"), ("e", "え"), ("o", "お")]),
    ("k", [("a", "か"), ("i", "き"), ("u", "く"), ("e", "け"), ("o", "こ")]),
    ("g", [("a", "が"), ("i", "ぎ"), ("u", "ぐ"), ("e", "げ"), ("o", "ご")]),
    ("s", [("a", "さ"), ("u", "す"), ("e", "せ"), ("o", "そ")]),
    ("sh", [("a", "しゃ"), ("i", "し"), ("u", "しゅ"), ("o", "しょ")]),
    ("z", [("a", "ざ"), ("u", "ず"), ("e", "ぜ"), ("o", "ぞ")]),
    ("j", [("a", "じゃ"), ("i", "じ"), ("u", "じゅ"), ("o", "じょ")]),
    ("t", [("a", "た"), ("e", "て"), ("o", "と")]),
    ("ch", [("a", "ちゃ"), ("i", "ち"), ("u", "ちゅ"), ("o", "ちょ")]),
    ("ts", [("u", "つ")]),
    ("d", [("a", "だ"), ("e", "で"), ("o", "ど")]),
    ("n", [("a", "な"), ("u", "ぬ"), ("e", "ね"), ("o", "の")]),
    ("ny", [("a", "にゃ"), ("i", "に"), ("u", "にゅ"), ("o", "にょ")]),
    ("h", [("a", "は"), ("e", "へ"), ("o", "ほ")]),
    ("hy", [("a", "ひゃ"), ("i", "ひ"), ("u", "ひゅ"), ("o", "ひょ")]),
    ("f", [("u", "ふ")]),
    ("b", [("a", "ば"), ("i", "び"), ("u", "ぶ"), ("e", "べ"), ("o", "ぼ")]),
    ("p", [("a", "ぱ"), ("i", "ぴ"), ("u", "ぷ"), ("e", "ぺ"), ("o", "ぽ")]),
    ("m", [("a", "ま"), ("i", "み"), ("u", "む"), ("e", "め"), ("o", "も")]),
    ("y", [("a", "や"), ("u", "ゆ"), ("o", "よ")]),
    ("r", [("a", "ら"), ("i", "り"), ("u", "る"), ("e", "れ"), ("o", "ろ")]),
    ("w", [("a", "わ")]),
    ("ky", [("a", "きゃ"), ("u", "きゅ"), ("o", "きょ")]),
    ("gy", [("a", "ぎゃ"), ("u", "ぎゅ"), ("o", "ぎょ")]),
    ("by", [("a", "びゃ"), ("u", "びゅ"), ("o", "びょ")]),
    ("py", [("a", "ぴゃ"), ("u", "ぴゅ"), ("o", "ぴょ")]),
    ("my", [("a", "みゃ"), ("u", "みゅ"), ("o", "みょ")]),
    ("ry", [("a", "りゃ"), ("u", "りゅ"), ("o", "りょ")]),
]

# Romaji spellings that do not follow cls+vowel.
_SPECIAL_KEY = {"し": "shi", "ち": "chi", "つ": "tsu", "ふ": "fu", "じ": "ji"}


def _build() -> dict[str, Mora]:
    out: dict[str, Mora] = {}
    for cls, vs in _ROWS:
        for v, kana in vs:
            if kana in _SPECIAL_KEY:
                key = _SPECIAL_KEY[kana]
            elif cls in ("sh", "ch", "j") and v != "i":
                key = f"{cls}{v}"
            elif cls in ("hy", "ny") and v == "i":
                key = f"{cls[0]}i"      # hi / ni
            else:
                key = f"{cls}{v}"
            out[key] = Mora(key, kana, cls, v)
    out["nn"] = Mora("nn", "ん", "n", "")
    return out


MORAE: dict[str, Mora] = _build()
_BY_CLS_V: dict[tuple[str, str], Mora] = {
    (m.cls, m.vowel): m for m in MORAE.values() if m.key != "nn"
}

# Closest-sounding fallbacks when a (class, vowel) pair has no mora of its own.
_FALLBACK = {
    ("s", "i"): ("sh", "i"), ("z", "i"): ("j", "i"), ("t", "i"): ("ch", "i"),
    ("d", "i"): ("j", "i"), ("t", "u"): ("ts", "u"), ("d", "u"): ("z", "u"),
    ("h", "i"): ("hy", "i"), ("h", "u"): ("f", "u"),
    ("f", "a"): ("h", "a"), ("f", "i"): ("hy", "i"), ("f", "e"): ("h", "e"),
    ("f", "o"): ("h", "o"), ("n", "i"): ("ny", "i"), ("k", "i"): ("k", "i"),
    ("sh", "e"): ("s", "e"), ("ch", "e"): ("t", "e"), ("j", "e"): ("z", "e"),
    ("ts", "a"): ("t", "a"), ("ts", "e"): ("t", "e"), ("ts", "o"): ("t", "o"),
}


def lookup(cls: str, vowel: str, palatal: bool = False) -> Mora | None:
    """Return the mora for a consonant class + vowel, applying palatalisation and fallbacks."""
    if palatal and cls and vowel in "auo" and (cls + "y", vowel) in _BY_CLS_V:
        return _BY_CLS_V[(cls + "y", vowel)]
    m = _BY_CLS_V.get((cls, vowel))
    if m:
        return m
    fb = _FALLBACK.get((cls, vowel))
    if fb:
        return _BY_CLS_V.get(fb)
    return None


VOICELESS_CLASSES = {"k", "s", "sh", "t", "ch", "ts", "h", "hy", "f", "p", "ky", "py"}

# Typical consonant lengths (ms) used when nothing better can be measured.
DEFAULT_PRE_MS = {
    "": 25, "k": 70, "g": 55, "s": 110, "sh": 120, "z": 65, "j": 85, "t": 65,
    "ch": 90, "ts": 90, "d": 55, "n": 60, "ny": 70, "h": 85, "hy": 95, "f": 95,
    "b": 55, "p": 65, "m": 60, "y": 60, "r": 45, "w": 55, "ky": 85, "gy": 70,
    "by": 70, "py": 80, "my": 75, "ry": 65,
}


def all_keys() -> list[str]:
    return list(MORAE.keys())


# --------------------------------------------------------------------------- IPA

_VOWELS = {
    "a": "a", "ɑ": "a", "ä": "a", "ɐ": "a", "æ": "a", "ʌ": "a", "ɒ": "a",
    "i": "i", "ɪ": "i", "ɨ": "i", "ᵻ": "i", "y": "i",
    "u": "u", "ʊ": "u", "ɯ": "u", "ʉ": "u",
    "e": "e", "ɛ": "e", "ø": "e", "œ": "e", "ɜ": "e",
    "o": "o", "ɔ": "o", "ɵ": "o",
}
# Diphthongs collapse to their first vowel.
_DIPH = {"aɪ": "a", "aʊ": "a", "eɪ": "e", "oʊ": "o", "ɔɪ": "o", "əʊ": "o", "ai": "a", "au": "a"}

_CONS = {
    "k": "k", "c": "k", "q": "k", "ɡ": "g", "g": "g", "ɣ": "g", "ɟ": "g",
    "s": "s", "θ": "s", "ʂ": "sh", "ʃ": "sh", "ɕ": "sh",
    "z": "z", "ð": "z", "ʒ": "j", "ʑ": "j", "dʒ": "j", "dʑ": "j", "dZ": "j", "tS": "ch",
    "tʃ": "ch", "tɕ": "ch", "ts": "ts",
    "t": "t", "d": "d", "ʈ": "t", "ɖ": "d",
    "n": "n", "ɲ": "ny", "m": "m",
    "h": "h", "x": "h", "χ": "h", "ħ": "h", "ç": "hy", "f": "f", "ɸ": "f",
    "b": "b", "β": "b", "v": "b", "p": "p",
    "w": "w", "ʋ": "w", "j": "y",
    "ɾ": "r", "r": "r", "l": "r", "ɹ": "r", "ʁ": "r", "ɭ": "r", "ɫ": "r", "ɽ": "r", "ɻ": "r",
}
_NASAL_CODA = {"ŋ", "ɴ", "N", "n̩", "m̩"}
_STRIP = re.compile(r"[ːː:̩̪̞̥̃̊ʰʷ^.˞0-9\[\]\"']")


@dataclass(frozen=True)
class Sym:
    kind: str            # 'V' vowel, 'C' consonant, 'G' glide, 'N' nasal coda, 'X' other
    value: str = ""      # vowel letter, consonant class
    palatal: bool = False


def classify(sym: str) -> Sym:
    """Classify one IPA token from the phoneme recogniser."""
    if sym in _DIPH:
        return Sym("V", _DIPH[sym])
    if sym in _NASAL_CODA:
        return Sym("N")
    palatal = "ʲ" in sym
    base = _STRIP.sub("", sym.replace("ʲ", ""))
    if not base:
        return Sym("X")
    if base in _DIPH:
        return Sym("V", _DIPH[base])
    if base == "j":
        return Sym("G")
    if base in _CONS:
        return Sym("C", _CONS[base], palatal)
    if base in _VOWELS:
        return Sym("V", _VOWELS[base])
    if base[0] in _VOWELS and len(base) <= 3:      # e.g. 'ɑɹ', 'ɛɹ': r-coloured vowel
        return Sym("V", _VOWELS[base[0]])
    return Sym("X")
