"""Add Mandarin (plain toneless pinyin) syllables to a bank that already has English CVVC units.

Whole syllables are built from existing clips, no RVC: initial+final from the closest English units, glides
(i-/u-/ü-) and nasal codas spliced on. Tones are not in the samples (they come from the pitch curve).
Approximations: ü -> "y"+"uw", retroflex zh/ch/sh/r -> jh/ch/sh/r (apical vowel -> "er"), j/q/x -> jh/ch/sh,
z/c -> "t"+"s". Where a plain pinyin spelling equals a plain romaji alias from crossling, pinyin wins.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import audio
from .crossling import Mapper, oto_line

# initial -> allowed finals (finals of j q x use the ü-forms v ve van vn; they are spelled u ue uan un)
_JQX = "i ia ie iao iu ian in iang ing iong v ve van vn"
_GKH = "a e ai ei ao ou an en ang eng ong u ua uo uai ui uan un uang"
_ZHCHSH = "a e ai ei ao ou an en ang eng ong i u ua uo uai ui uan un uang"
_ZCS = "a e ai ei ao ou an en ang eng ong i u uo ui uan un"
INITIALS: dict[str, str] = {
    "b": "a o ai ei ao an en ang eng i ie iao ian in ing u",
    "p": "a o ai ei ao ou an en ang eng i ie iao ian in ing u",
    "m": "a o e ai ei ao ou an en ang eng i ie iao iu ian in ing u",
    "f": "a o ei ou an en ang eng u",
    "d": "a e ai ei ao ou an en ang eng ong i ia ie iao iu ian ing u uo ui uan un",
    "t": "a e ai ei ao ou an ang eng ong i ie iao ian ing u uo ui uan un",
    "n": "a e ai ei ao ou an en ang eng ong i ie iao iu ian in iang ing u uo uan v ve",
    "l": "a e ai ei ao ou an ang eng ong i ia ie iao iu ian in iang ing u uo uan un v ve",
    "g": _GKH, "k": _GKH, "h": _GKH,
    "j": _JQX, "q": _JQX, "x": _JQX,
    "zh": _ZHCHSH, "ch": _ZHCHSH, "sh": _ZHCHSH,
    "r": "e ao ou an en ang eng ong i u ua uo ui uan un",
    "z": _ZCS, "c": _ZCS, "s": _ZCS,
}
# syllables without an initial: spelling -> final
ZERO: dict[str, str] = {
    "a": "a", "o": "o", "e": "e", "ai": "ai", "ei": "ei", "ao": "ao", "ou": "ou", "an": "an", "en": "en",
    "ang": "ang", "eng": "eng", "er": "er", "yi": "i", "ya": "ia", "ye": "ie", "yao": "iao", "you": "iu",
    "yan": "ian", "yin": "in", "yang": "iang", "ying": "ing", "yong": "iong", "yu": "v", "yue": "ve",
    "yuan": "van", "yun": "vn", "wu": "u", "wa": "ua", "wo": "uo", "wai": "uai", "wei": "ui", "wan": "uan",
    "wen": "un", "wang": "uang", "weng": "ueng",
}
# final -> (glide, vowel, coda)
FINALS: dict[str, tuple[str, str, str]] = {
    "a": ("", "aa", ""), "o": ("", "ao", ""), "e": ("", "ah", ""), "ai": ("", "ay", ""), "ei": ("", "ey", ""),
    "ao": ("", "aw", ""), "ou": ("", "ow", ""), "an": ("", "aa", "n"), "en": ("", "ah", "n"),
    "ang": ("", "aa", "ng"), "eng": ("", "ah", "ng"), "ong": ("", "uh", "ng"), "er": ("", "er", ""),
    "i": ("", "iy", ""), "ia": ("y", "aa", ""), "ie": ("y", "eh", ""), "iao": ("y", "aw", ""),
    "iu": ("y", "ow", ""), "ian": ("y", "eh", "n"), "in": ("", "iy", "n"), "iang": ("y", "aa", "ng"),
    "ing": ("", "iy", "ng"), "iong": ("y", "uh", "ng"),
    "u": ("", "uw", ""), "ua": ("w", "aa", ""), "uo": ("w", "ao", ""), "uai": ("w", "ay", ""),
    "ui": ("w", "ey", ""), "uan": ("w", "aa", "n"), "un": ("w", "ah", "n"), "uang": ("w", "aa", "ng"),
    "ueng": ("w", "ah", "ng"),
    "v": ("y", "uw", ""), "ve": ("y", "eh", ""), "van": ("y", "eh", "n"), "vn": ("y", "iy", "n"),
}
CONS = {"b": ["b"], "p": ["p"], "m": ["m"], "f": ["f"], "d": ["d"], "t": ["t"], "n": ["n"], "l": ["l"],
        "g": ["g"], "k": ["k"], "h": ["hh"], "j": ["jh"], "q": ["ch"], "x": ["sh"], "zh": ["jh"],
        "ch": ["ch"], "sh": ["sh"], "r": ["r", "zh"], "s": ["s"]}
VOWEL_FALLBACK = {"aa": ["aa", "ah", "ae"], "ao": ["ao", "ow", "aa"], "ah": ["ah", "aa"], "ay": ["ay"],
                  "ey": ["ey", "eh"], "aw": ["aw", "ao"], "ow": ["ow", "ao"], "iy": ["iy", "ih"],
                  "uw": ["uw", "uh"], "eh": ["eh", "ey"], "uh": ["uh", "uw"], "ih": ["ih", "iy"],
                  "er": ["er", "ah"]}


def spell(ini: str, fin: str) -> str:
    f = fin.replace("v", "u") if ini in ("j", "q", "x") else fin
    return ini + f


def syllables() -> dict[str, tuple[str, str]]:
    """pinyin spelling -> (initial, final); '' initial for zero-initial syllables."""
    out = {name: ("", fin) for name, fin in ZERO.items()}
    for ini, fins in INITIALS.items():
        for fin in fins.split():
            out[spell(ini, fin)] = (ini, fin)
    return out


class _Syl:
    def __init__(self, m: Mapper):
        self.m = m

    def unit(self, kind: str, a: str, b: str = "") -> str | None:
        """Existing unit key for a vowel/consonant name, trying vowel fall-backs; prefers recordings."""
        cands = []
        for v in VOWEL_FALLBACK.get(b if kind == "cv" else a, [b if kind == "cv" else a]):
            cands.append(f"cv_{a}_{v}" if kind == "cv" else f"vc_{v}_{b}")
        cands = [k for k in cands if k in self.m.have]
        return next((k for k in cands if k in self.m.recorded), cands[0] if cands else None)

    def spec(self, ini: str, fin: str):
        glide, vowel, coda = FINALS[fin]
        if fin == "i" and ini in ("zh", "ch", "sh", "r"):
            vowel = "er"                                       # apical vowel ~ r-coloured
        elif fin == "i" and ini in ("z", "c", "s"):
            vowel = "ih"
        onset_v = {"": vowel, "y": "iy", "w": "uw"}[glide]
        pieces: list[tuple[str | None, str]] = []
        if ini == "":
            if glide:
                pieces.append((self.unit("cv", glide, vowel), "full"))
            else:
                pieces.append((next((f"v_{v}" for v in VOWEL_FALLBACK[vowel] if f"v_{v}" in self.m.have), None), "full"))
        elif ini in ("z", "c"):                                # ts = "t" burst + "s"
            pieces.append((self.unit("cv", "t", onset_v), "cons"))
            if glide:
                pieces += [(self.unit("cv", "s", onset_v), "cons"), (self.unit("cv", glide, vowel), "full")]
            else:
                pieces.append((self.unit("cv", "s", vowel), "full"))
        else:
            c = next((x for x in CONS[ini] if self.unit("cv", x, onset_v)), CONS[ini][0])
            if glide:
                pieces += [(self.unit("cv", c, onset_v), "cons"), (self.unit("cv", glide, vowel), "full")]
            else:
                pieces.append((self.unit("cv", c, vowel), "full"))
        if coda:
            pieces.append((self.unit("vc", vowel, coda), "tail"))
        return pieces


def add_mandarin(bank: Path) -> dict:
    """Add pinyin syllables to `bank` (modified in place). Returns a summary."""
    m = Mapper(bank)
    syl = _Syl(m)
    lines = (bank / "oto.ini").read_bytes().decode("cp932").splitlines()
    existing = {l.split("=", 1)[1].split(",")[0]: i for i, l in enumerate(lines)}
    summary = {"syllables": 0, "direct": 0, "spliced": 0, "missing": [], "recorded_backed": 0, "rvc_backed": 0,
               "romaji_replaced_by_pinyin": []}
    drop: set[int] = set()
    new: list[str] = []
    for name, (ini, fin) in sorted(syllables().items()):
        spec = syl.spec(ini, fin)
        if any(k is None for k, _ in spec):
            summary["missing"].append(name)
            continue
        keys = [k for k, _ in spec]
        if len(spec) == 1:                                     # plain unit: alias the existing wav
            wav, params = keys[0], m.oto[keys[0]][1]
            summary["direct"] += 1
        else:
            y, params = m.build(spec)
            wav = f"zh_{name}"
            audio.write_wav(bank / f"{wav}.wav", y, audio.SR_BANK)
            summary["spliced"] += 1
        summary["recorded_backed" if all(k in m.recorded for k in keys) else "rvc_backed"] += 1
        summary["syllables"] += 1
        if name in existing:                                   # same spelling as a Japanese romaji alias
            old = lines[existing[name]]
            if old.split("=")[0][:-4] == wav:                   # same sample: nothing to do
                continue
            summary["romaji_replaced_by_pinyin"].append(name)
            drop.add(existing[name])
        new.append(oto_line(wav, name, params))
    out = [l for i, l in enumerate(lines) if i not in drop] + new
    (bank / "oto.ini").write_bytes(("\n".join(out) + "\n").encode("cp932"))
    rep_path = bank / "voice2utau_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    rep["mandarin"] = {**{k: v for k, v in summary.items()}, "aliases": "plain pinyin, no tone numbers"}
    rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = bank / "readme.txt"
    readme.write_text(readme.read_text(encoding="utf-8") +
                      "Mandarin: plain toneless pinyin aliases (ma, shi, zhuang, lv, nve ...), built from the English recordings, so\n"
                      "Mandarin carries the English accent; syllables with glides/endings are spliced (files zh_*.wav). Tones come from\n"
                      "the pitch curve. Where a plain spelling is both a pinyin syllable and a Japanese romaji alias, pinyin is used;\n"
                      "the Japanese sound stays available by its hiragana name (see voice2utau_report.json: mandarin.romaji_replaced_by_pinyin).\n",
                      encoding="utf-8")
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Add Mandarin pinyin syllables to a finished bank (in place).")
    ap.add_argument("bank", type=Path)
    a = ap.parse_args()
    s = add_mandarin(a.bank)
    print(f"{s['syllables']} syllables: {s['direct']} direct, {s['spliced']} spliced, missing {s['missing']}; "
          f"recorded-backed {s['recorded_backed']}, RVC-backed {s['rvc_backed']}")
    print(f"pinyin replaced {len(s['romaji_replaced_by_pinyin'])} Japanese romaji aliases: {s['romaji_replaced_by_pinyin']}")


if __name__ == "__main__":
    main()
