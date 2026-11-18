"""Rigo: a female voicebank made of Apple sounds (iOS, macOS, Apple Watch, AirPods). Minimal and cool.

Voice: the cleanest, glassiest Apple alert tones (Glass, Ping, Submarine, Hero, Purr, Note, Chord ...) - the three
richest steady ones are blended - with a real voice layered under them so the vowels read. Consonant hiss from Apple's
whooshes (Mail sent, Swish, shake), k/t/p releases from the iPhone keyboard click, Tink/Tock, the lock and the shutter.
Signature: a faint haptic tap (a short low thump, like the Taptic Engine) at the start of each syllable.
Japanese CV bank (synthbank) and English Arpasing bank (arpasing) from the same recipe; every clip voice-screened.

    python -m voice2utau.rigo japanese OUT [--sfx-bank DIR]
    python -m voice2utau.arpasing build rigo OUT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import arpasing, ninchan, synthbank
from .characters import _hooks, _merge_sfx

SR = synthbank.SR
F0 = 233.08                        # A#3: a little lower and cooler than MADO's C4
SCALE = 0.98
CLEAN = Path(__file__).resolve().parent.parent / "voicebanks" / "rigo" / "clean"
TONES = ["mac/osx_Glass", "mac/osx_Ping", "mac/osx_Submarine", "mac/osx_Hero", "mac/osx_Purr", "mac/macos_Note",
         "mac/macos_Chord", "mac/macos_Pulse", "mac/macos_Input", "mac/mojave_Lighthouse", "ios/ios17_Crystals",
         "ios/ios17_Radiate"]
HISS = ["mac/sb_Mail_sent_whoosh", "mac/macos_Swish", "ios/ios41_mail_sent", "mac/macos_tweet_sent", "ios/ios41_shake"]
CLICKS = ["ios/ios421_Tock", "ios/ios421_Tink", "ios/ios41_lock", "ios/ios41_photoShutter", "mac/classic_Single_Click",
          "ios/sb_OG_iPhone_Keyboard_Sound", "ios/sb_IPhone_Original_Keyboard_Sound", "ios/sb_Iphone_Keyboard_Click_SFX"]


def _files(names: list[str], root: Path = CLEAN) -> list[Path]:
    return [p for p in (root / f"{n}.wav" for n in names) if p.exists()]


def haptic(level_db: float = -20.0, hz: float = 170.0, ms: float = 16.0):
    """A Taptic-style tap: a few cycles of a low sine with a fast decay, just under the vowel."""
    n = int(ms / 1000 * SR)
    t = np.arange(n) / SR
    tap = np.sin(2 * np.pi * hz * t) * np.exp(-t / (ms / 1000 / 4))

    def post(y: np.ndarray, c: int, rng) -> tuple[np.ndarray, int]:
        y = np.asarray(y, dtype=np.float64).copy()
        vow = y[c + int(0.06 * SR): c + int(0.2 * SR)]
        ref = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
        a = max(0, c - int(0.003 * SR))
        k = min(n, len(y) - a)
        y[a:a + k] += tap[:k] * ref * np.sqrt(2) * 10 ** (level_db / 20)
        return y, c
    return post


def parts(root: Path = CLEAN):
    old = synthbank.F0
    synthbank.F0 = F0
    try:
        tones = ninchan.pick_carriers([], top=3, keep=0.5, files=_files(TONES, root))
    finally:
        synthbank.F0 = old
    if not tones:
        raise ValueError("no steady Apple tone found for Rigo's voice")
    hiss = _files(HISS, root)
    clicks = _files(CLICKS, root)
    return (ninchan.make_source([t[3] for t in tones]), ninchan.make_noise(hiss, mix=0.6) if hiss else None,
            ninchan.make_burst(clicks) if clicks else None, [Path(t[0]).stem for t in tones])


def recipe(root: Path = CLEAN) -> arpasing.Recipe:
    src, noise, burst, tones = parts(root)
    return arpasing.Recipe("Rigo_EN", f0=F0, scale=SCALE, source=src, noise=noise, burst=burst, onset_post=haptic(),
                           vowel_mix=0.5,
                           readme=f"Rigo in English: minimal and cool. Apple alert tones ({', '.join(tones)}) with a "
                                  "real voice layered under them, Apple whooshes for the hiss, iPhone keyboard / "
                                  "Tink / Tock / lock clicks as releases, a faint haptic tap on each syllable.")


def build_japanese(out: Path, sfx_bank: Path | None = None, name: str = "Rigo", root: Path = CLEAN) -> Path:
    src, noise, burst, tones = parts(root)
    tap = haptic()
    old = synthbank.F0, synthbank.FORMANT_SCALE
    synthbank.F0, synthbank.FORMANT_SCALE = F0, SCALE
    try:
        with _hooks(arpasing.layered(src, 0.5), noise, burst, lambda y, c, m, rng: tap(y, c, rng)):
            bank = synthbank.build(out, name)
    finally:
        synthbank.F0, synthbank.FORMANT_SCALE = old
    n = _merge_sfx(bank, sfx_bank)
    (bank / "readme.txt").write_text(
        f"{name}\nA female Japanese CV voicebank made of Apple sounds (iOS, macOS, Apple Watch, AirPods). Minimal and "
        f"cool: Apple alert tones ({', '.join(tones)}) with a real voice layered under them, Apple whooshes for the "
        "hiss, iPhone keyboard / Tink / Tock / lock clicks as k/t/p releases, a faint haptic tap on each syllable. "
        f"Same synthesiser as MADO.\n{n} one-shot Apple sounds as <ios|mac|watch>_<name>.\nNo recorded voice: every "
        "clip was screened (Siri, VoiceOver and macOS speech samples are out).\nRender with voice2utau.sing --engine "
        "psola.\nThe source sounds belong to Apple. Personal use only.\n", encoding="utf-8")
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
