"""Target-language profiles: which units a voicebank contains and how to find them in speech."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import english, extract, morae
from .morae import Mora


@dataclass(frozen=True)
class Profile:
    code: str
    name: str
    units: dict[str, Mora]
    build_candidates: Callable
    group_of: Callable[[Mora], str]
    description: str
    default_min_conf: float = 0.45   # recogniser confidence needed to keep a candidate
    default_validate: bool = True     # acoustic cross-check of labels (see validate.py)
    native_lang: str | None = None    # recordings in another language are fallbacks (None = no notion of one)


PROFILES: dict[str, Profile] = {
    "ja": Profile("ja", "Japanese – hiragana CV (101 units)", morae.MORAE,
                  lambda phones, src, lang="en": extract.build_candidates(phones, src),
                  lambda m: m.cls, "Single-mora hiragana bank: か, きゃ, ん …"),
    "en": Profile("en", f"English – ARPAbet CVVC ({len(english.UNITS)} units)", english.UNITS,
                  english.build_candidates, english.group_of,
                  "CV 'k ae', VC 'ae t' and initial-vowel '- ae' units",
                  # measured on real English speech: confidence is the better filter here;
                  # the MFCC cross-check removed half the units for +2 points of precision
                  default_min_conf=0.7, default_validate=False, native_lang="en"),
}


def get(code: str) -> Profile:
    try:
        return PROFILES[code]
    except KeyError:
        raise ValueError(f"unknown language {code!r}; choose one of {sorted(PROFILES)}") from None
