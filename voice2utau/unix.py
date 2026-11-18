"""UNIX: a male voicebank made of Unix workstation sounds - NeXTSTEP 3.3 (taken straight off its disk), Sun SunOS /
Solaris demo sounds (/usr/demo/SOUND) and SGI IRIX 6.5 desktop (4Dwm) sounds. Old-school terminal: dry and precise.
No Linux (that is another voicebank), and NeXTSTEP rather than modern macOS (Rigo has those).

Voice: IRIX and Sun tones blended (ting, high.chink, computer, cowbell) with a real voice under them so the vowels read,
and a faint raw layer of Sun's 8 kHz "computer" beeps on the vowels. Hiss from IRIX whisk / soft crash and Sun flush /
crash; k/t/p releases from NeXT Tink / Pop / Bonk / Block and Sun's touch-tone keypad. Signature: 8-bit mu-law grit,
the sound of a Sun workstation's original /dev/audio. Every clip voice-screened (NeXT's spoken printer alerts, Ni,
Poit, the Graham-Bell quote, Sun's sample.au and the laughs are out).

    python -m voice2utau.unix japanese OUT [--sfx-bank DIR]
    python -m voice2utau.arpasing build unix OUT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import arpasing, ninchan, synthbank
from .characters import _hooks, _merge_sfx

SR = synthbank.SR
F0 = 146.83                        # D3: a precise baritone, above Puresu (B2) and ANDRO (C3)
SCALE = 0.9
JP_SCALE = 0.9
KEEP, VOWEL_MIX, DIRECT_DB = 0.8, 0.4, -20.0
MU_BITS = 8
CLEAN = Path(__file__).resolve().parent.parent / "voicebanks" / "unix" / "stage"
TONES = ["irix/ting", "irix/high_chink", "sun/computer", "sun/cowbell"]
DIRECT = ["sun/computer"]
HISS = ["irix/whisk", "irix/soft_crash", "irix/kerchick_echo", "sun/flush", "sun/crash", "next/Grab_OpenShutter"]
CLICKS = ["next/Tink", "next/Pop", "next/Bonk", "next/Block", "next/Funk", "sun/touchtone_1", "sun/touchtone_5",
          "sun/touchtone_9", "irix/chick"]


def _files(names: list[str], root: Path = CLEAN) -> list[Path]:
    return [p for p in (root / f"{n}.wav" for n in names) if p.exists()]


def mulaw(bits: int = MU_BITS, mu: float = 255.0):
    """Compand to `bits`-bit mu-law and back (Sun /dev/audio was 8-bit mu-law): grit that follows the signal."""
    q = 2 ** (bits - 1)

    def f(y: np.ndarray) -> np.ndarray:
        y = np.clip(np.asarray(y, dtype=np.float64), -1, 1)
        c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
        c = np.round(c * q) / q
        return np.sign(c) * np.expm1(np.abs(c) * np.log1p(mu)) / mu
    return f


def parts(root: Path = CLEAN, keep: float = KEEP):
    old = synthbank.F0
    synthbank.F0 = F0
    try:
        tones = ninchan.pick_carriers([], top=len(TONES), keep=keep, files=_files(TONES, root))
        raw = ninchan.pick_carriers([], top=len(DIRECT), keep=1.0, files=_files(DIRECT, root))
    finally:
        synthbank.F0 = old
    if not tones:
        raise ValueError("no steady Unix tone found for UNIX's voice")
    hiss, clicks = _files(HISS, root), _files(CLICKS, root)
    return (ninchan.make_source([t[3] for t in tones]), ninchan.make_noise(hiss, mix=0.55) if hiss else None,
            ninchan.make_burst(clicks) if clicks else None, [Path(t[0]).stem for t in tones],
            ninchan.make_source([t[3] for t in raw]) if raw else None)


def recipe(root: Path = CLEAN, keep: float = KEEP, vowel_mix: float = VOWEL_MIX, direct_db: float | None = DIRECT_DB,
           scale: float = SCALE, bits: int | None = MU_BITS) -> arpasing.Recipe:
    src, noise, burst, tones, raw = parts(root, keep)
    return arpasing.Recipe("UNIX_EN", f0=F0, scale=scale, source=src, noise=noise, burst=burst,
                           all_post=mulaw(bits) if bits else None,
                           vowel_mix=vowel_mix, direct=raw if direct_db is not None else None,
                           direct_db=direct_db if direct_db is not None else 0.0,
                           readme=f"UNIX in English: old-school terminal, dry and precise. IRIX / Sun tones "
                                  f"({', '.join(tones)}) with a real voice layered under them, a faint layer of Sun's "
                                  "8 kHz computer beeps on the vowels, IRIX / Sun hiss, NeXTSTEP clicks and Sun "
                                  "touch-tones as releases, 8-bit mu-law grit like a Sun workstation's /dev/audio.")


def build_japanese(out: Path, sfx_bank: Path | None = None, name: str = "UNIX", root: Path = CLEAN) -> Path:
    src, noise, burst, tones, raw = parts(root, KEEP)
    layered = arpasing.layered(src, VOWEL_MIX)
    g = 10 ** (DIRECT_DB / 20)
    grit = mulaw()

    def post(y, c, m, rng):                                  # the raw beeps on the vowel, after the filter, then grit
        n = len(y) - c
        y = np.asarray(y, dtype=np.float64).copy()
        if raw is not None and n > int(0.05 * SR):
            vow = y[c + int(0.06 * SR): c + int(0.25 * SR)]
            ref = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
            d = raw(n, rng)
            env = np.clip(np.arange(n) / (0.015 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.04 * SR), 0, 1)
            y[c:] += d * env / max(1e-9, float(np.sqrt((d ** 2).mean()))) * ref * g
        pk = float(np.abs(y).max()) or 1.0
        return grit(y / pk) * pk, c
    old = synthbank.F0, synthbank.FORMANT_SCALE
    synthbank.F0, synthbank.FORMANT_SCALE = F0, JP_SCALE
    try:
        with _hooks(layered, noise, burst, post):
            bank = synthbank.build(out, name)
    finally:
        synthbank.F0, synthbank.FORMANT_SCALE = old
    n = _merge_sfx(bank, sfx_bank)
    (bank / "readme.txt").write_text(
        f"{name}\nA male Japanese CV voicebank made of Unix workstation sounds: NeXTSTEP 3.3, Sun SunOS / Solaris and "
        f"SGI IRIX. Old-school terminal, dry and precise: IRIX / Sun tones ({', '.join(tones)}) with a little real "
        "voice under them, a faint layer of Sun's 8 kHz computer beeps on the vowels, IRIX / Sun hiss, NeXTSTEP clicks "
        "and Sun touch-tones as k/t/p releases, 8-bit mu-law grit. Same synthesiser as MADO.\n"
        f"{n} one-shot Unix sounds as <next|sun|irix>_<name>.\nNo recorded voice: every clip was screened (NeXT's "
        "spoken printer alerts, Ni, Poit and the Graham-Bell quote, Sun's sample.au and the laughs are out).\nNo "
        "Linux.\nRender with voice2utau.sing --engine psola.\nThe source sounds belong to their makers (NeXT / Apple, "
        "Sun / Oracle, SGI). Personal use only.\n", encoding="utf-8")
    return bank


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("japanese")
    j.add_argument("out", type=Path)
    j.add_argument("--sfx-bank", type=Path)
    a = ap.parse_args()
    print("wrote", build_japanese(a.out, a.sfx_bank))


if __name__ == "__main__":
    main()
