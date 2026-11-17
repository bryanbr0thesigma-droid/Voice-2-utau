"""Convert a VCCV English UST (VDR / Cz-style aliases such as "-sO", "n sh", "sh3", "uk-") into Arpasing notes, so an
Arpasing bank (voice2utau.arpasing) can sing it with the UST's own timing.

Each VCCV alias is read as a phone string (start '-', end '-', cluster '_', space-separated or run together), mapped to
ARPAbet, and becomes one or more Arpasing units ('- V', 'C V', 'V C', 'C C', 'V V', 'V -', 'C -'). A note that needs
two units (e.g. "uk-" = "ah k" + "k -") is split, the vowel-ending unit getting the larger share.
"""
from __future__ import annotations

import copy

from .sing import SILENT, Note

# VDR VCCV vowel symbols -> ARPAbet (read off the lyrics: bA = "ba(by)", mE = "me", sO = "so", f9 9l = "foul" ...)
VOWELS = {"a": ["aa"], "@": ["ae"], "&": ["ae"], "A": ["ey"], "e": ["eh"], "E": ["iy"], "i": ["ih"], "I": ["ay"],
          "o": ["uw"], "O": ["ow"], "u": ["ah"], "6": ["uh"], "3": ["er"], "8": ["ao"], "0": ["ao"], "9": ["aw"],
          "x": ["ax"], "U": ["uh"], "Q": ["aw"], "Y": ["ay"]}
CONS = {"b": ["b"], "ch": ["ch"], "d": ["d"], "dd": ["d"], "dh": ["dh"], "f": ["f"], "g": ["g"], "h": ["hh"],
        "hh": ["hh"], "j": ["jh"], "k": ["k"], "l": ["l"], "ll": ["ax", "l"], "m": ["m"], "mm": ["ax", "m"],
        "n": ["n"], "nn": ["ax", "n"], "ng": ["ng"], "p": ["p"], "r": ["r"], "s": ["s"], "sh": ["sh"], "t": ["t"],
        "tt": ["t"], "th": ["th"], "v": ["v"], "w": ["w"], "y": ["y"], "z": ["z"], "zh": ["zh"]}
SPECIAL = {"1ng": ["ih", "ng"]}
SYMBOLS = sorted(list(VOWELS) + list(CONS) + list(SPECIAL), key=len, reverse=True)
ARPA_VOWELS = {"aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}


def tokens(s: str) -> list[str] | None:
    out, i = [], 0
    while i < len(s):
        for sym in SYMBOLS:
            if s.startswith(sym, i):
                out += SPECIAL.get(sym) or VOWELS.get(sym) or CONS[sym]
                i += len(sym)
                break
        else:
            return None
    return out


def parse(alias: str) -> tuple[bool, list[str], bool] | None:
    """(starts from silence, ARPAbet phones, ends in silence) for one VCCV alias, or None if unreadable."""
    a = alias.strip()
    start = a.startswith("-")
    end = a.endswith("-") and len(a) > 1
    a = a.strip("-").lstrip("_")
    phones: list[str] = []
    for part in a.split():
        t = tokens(part)
        if t is None:
            return None
        for p in t:                                        # "1ng ng", "Ar r": a doubled consonant is one consonant
            if not (phones and p == phones[-1] and p not in ARPA_VOWELS):
                phones.append(p)
    return (start, phones, end) if phones else None


def units(start: bool, phones: list[str], end: bool, prev_last: str | None) -> list[str]:
    us = []
    if len(phones) == 1 and not start and not end:
        p = phones[0]
        if p in ARPA_VOWELS and prev_last:                 # a lone vowel continues from the previous sound
            return [f"{prev_last} {p}" if prev_last != p else f"{p} {p}"]
        return [f"- {p}"] if p in ARPA_VOWELS else [f"{p} -"]
    if start and phones[0] in ARPA_VOWELS:
        us.append(f"- {phones[0]}")
    for a, b in zip(phones, phones[1:]):
        us.append(f"{a} {b}")
    if end:
        us.append(f"{phones[-1]} -")
    if not us:                                             # "-C" alone
        us.append(f"- {phones[0]}")
    return us


def convert(notes: list[Note], oto: dict | None = None) -> tuple[list[Note], list[str]]:
    """VCCV notes -> Arpasing notes (split where needed). Returns (notes, aliases that were not understood)."""
    out, bad, prev_last = [], [], None
    for n in notes:
        if n.lyric in SILENT:
            out.append(n)
            prev_last = None
            continue
        p = parse(n.lyric)
        if p is None:
            bad.append(n.lyric)
            m = copy.copy(n)
            m.lyric = "R"
            out.append(m)
            continue
        start, phones, end = p
        us = units(start, phones, end, prev_last)
        if oto is not None:                                # drop units the bank lacks (it has the full set, normally)
            us = [u for u in us if u in oto] or us[:1]
        w = [3.0 if u.split()[-1] in ARPA_VOWELS else 1.0 for u in us]
        tot = sum(w)
        t0, start_s = 0, n.start
        for k, (u, wk) in enumerate(zip(us, w)):
            m = copy.copy(n)
            m.lyric = u
            m.pre = m.ovl = None
            if len(us) > 1:
                ln = n.length - t0 if k == len(us) - 1 else max(1, int(round(n.length * wk / tot)))
                m.length = ln
                m.dur = n.dur * ln / n.length
                m.start = start_s
                start_s += m.dur
                t0 += ln
                m.envelope = []
                if k > 0:
                    m.pbw, m.pby, m.pbs, m.vbr = [], [], "", []
            out.append(m)
        prev_last = None if end else phones[-1]
    return out, bad
