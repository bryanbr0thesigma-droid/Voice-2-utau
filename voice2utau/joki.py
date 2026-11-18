"""Joki (蒸気, "steam"): a male voicebank made of Steam sounds - SteamOS, Steam Deck (LCD and OLED), Steam Machine, Steam
Frame, Big Picture, the Steam desktop app and the Steam website's chat. Chill and cozy; a British lad in his late teens,
and the British accent carries into his Japanese.

Voice: the startup tones of five Steam devices blended (many devices, one lad) over a relaxed, lightly airy teenage
voice. Hiss from the Deck's side-menu / tab / modal swooshes; k/t/p releases from the Deck's typing and slider ticks
and the screenshot click. Signature: Steam's soft pop at the start of every syllable.
Accent: English - non-rhotic r (car, word: no r), British vowels (GOAT as [əʊ], back [ɑː], rounded THOUGHT, fronted
GOOSE, open TRAP and STRUT, [ɜː] for NURSE). Japanese - the same vowel colours on a/e/o/u, an English [ɹ] instead of
the Japanese flap, and English-style aspirated p/t/k, like a British lad singing in Japanese.

    python -m voice2utau.joki japanese OUT [--sfx-bank DIR]
    python -m voice2utau.arpasing build joki OUT
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from . import arpasing, audio, ninchan, synthbank
from .characters import _hooks, _merge_sfx
from .linx import blend, young_voice

SR = synthbank.SR
F0 = 155.56                        # D#3: a lighter, younger voice than the grown men (ANDRO C3 .. UNIX D3)
SCALE = 0.91                       # a late-teen vocal tract: a touch shorter than an adult man's
JP_SCALE = 0.91
KEEP, VOWEL_MIX = 0.75, 0.6
BREATH, TILT = 0.1, 0.45           # relaxed and soft (chill), a little air - less than LINX's breathy warmth
POP_DB = -21.0
CLEAN = Path(__file__).resolve().parent.parent / "voicebanks" / "joki" / "stage"
TONES = ["deck/startup_oled", "frame/startup", "steamos/family_startup", "machine/startup", "bigpicture/startup"]
HISS = ["deck/side_menu_fly_in", "deck/side_menu_fly_out", "deck/tab_transition_01", "deck/switch_toggle_off",
        "deck/switch_toggle_on", "deck/hide_modal"]
CLICKS = ["deck/typing", "deck/slider_down", "deck/slider_up", "desktop/camera1", "deck/default_activation"]
POP = "desktop/pop_sound"

# British English (southern, young) against the American reference table, female-reference Hz (FORMANT_SCALE applies)
BRITISH_EN = {
    "ow": ((560, 1450, 2800), (470, 1150, 2700)),      # GOAT [əʊ]: starts central
    "aa": ((850, 1250, 2700), None),                  # PALM / LOT: back [ɑː]
    "ao": ((560, 880, 2600), None),                   # THOUGHT: high, rounded [ɔː]
    "uw": ((420, 1650, 2400), None),                  # GOOSE: fronted [ʉː]
    "uh": ((520, 1400, 2700), None),                  # FOOT: fronted, unrounded
    "ae": ((900, 1650, 2800), None),                  # TRAP: open [a]
    "ah": ((800, 1450, 2800), None),                  # STRUT: [ɐ]
    "er": ((580, 1600, 2700), None),                  # NURSE: [ɜː], no r colour
    "ay": ((850, 1300, 2750), (480, 2300, 3000)),      # PRICE: backer start
    "aw": ((900, 1650, 2850), (520, 1100, 2700)),      # MOUTH: fronter start
    "ey": ((620, 2200, 2900), (440, 2650, 3200)),      # FACE: lower start
}
# Japanese vowels with British colour: back a, rounded o, rounded fronted u (young British GOOSE), open e
# (milder than his English: the full British set - o (560, 880), u F2 1550, e (600, 2050) - pushed o toward u and lost e;
# Japanese vowels scored 54/101 against 73 with no accent; this colour keeps 65)
BRITISH_JA_VOWELS = {"a": (800, 1220, 2700), "o": (640, 920, 2600), "u": (340, 1450, 2350), "e": (580, 2150, 2850)}
BRITISH_JA_R = (400, 1150, 1650)                       # English [ɹ]: low F3, in place of the flap


def _files(names: list[str], root: Path = CLEAN) -> list[Path]:
    return [p for p in (root / f"{n}.wav" for n in names) if p.exists()]


@contextlib.contextmanager
def british_japanese():
    """A British accent on the Japanese CV synthesiser: vowel colours, [ɹ] for the flap, aspirated p/t/k."""
    saved = dict(synthbank.VOWELS), dict(synthbank.CLASSES), dict(synthbank.GLIDE_START)
    synthbank.VOWELS.update(BRITISH_JA_VOWELS)
    synthbank.GLIDE_START["r"] = BRITISH_JA_R
    synthbank.CLASSES["r"] = ("glide", "r", 0, 50, 70, {})
    synthbank.CLASSES["ry"] = ("glide", "r", 0, 55, 80, {})
    for c, vot in (("k", 62), ("t", 55), ("p", 48), ("ky", 64), ("py", 50)):
        kind, place, cl, cm, tr, ex = synthbank.CLASSES[c]
        synthbank.CLASSES[c] = (kind, place, cl, cm + (vot - ex["vot"]), tr, {**ex, "vot": vot})
    try:
        yield
    finally:
        for d, old in zip((synthbank.VOWELS, synthbank.CLASSES, synthbank.GLIDE_START), saved):
            d.clear(); d.update(old)


def pop(root: Path = CLEAN, level_db: float = POP_DB, ms: float = 60.0):
    """Steam's soft pop, short and quiet, at the start of each syllable."""
    f = _files([POP], root)
    if not f:
        return lambda y, c, rng: (y, c)
    x, sr = audio.read_wav(f[0])
    x = np.asarray(audio.resample(x, sr, SR) if sr != SR else x, dtype=np.float64)
    a = int(np.argmax(np.abs(x) > 0.2 * np.abs(x).max()))
    t = x[max(0, a - int(0.002 * SR)): a + int(ms / 1000 * SR)].copy()
    t *= np.clip((len(t) - np.arange(len(t))) / (0.02 * SR), 0, 1)
    t /= max(1e-9, float(np.sqrt((t ** 2).mean())))

    def post(y: np.ndarray, c: int, rng) -> tuple[np.ndarray, int]:
        y = np.asarray(y, dtype=np.float64).copy()
        vow = y[c + int(0.06 * SR): c + int(0.2 * SR)]
        ref = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
        s = max(0, c - int(0.005 * SR))
        k = min(len(t), len(y) - s)
        y[s:s + k] += t[:k] * ref * 10 ** (level_db / 20)
        return y, c
    return post


def parts(root: Path = CLEAN, keep: float = KEEP):
    old = synthbank.F0
    synthbank.F0 = F0
    try:
        tones = ninchan.pick_carriers([], top=len(TONES), keep=keep, files=_files(TONES, root))
    finally:
        synthbank.F0 = old
    if not tones:
        raise ValueError("no steady Steam tone found for Joki's voice")
    hiss, clicks = _files(HISS, root), _files(CLICKS, root)
    return (ninchan.make_source([t[3] for t in tones]), ninchan.make_noise(hiss, mix=0.55) if hiss else None,
            ninchan.make_burst(clicks) if clicks else None,
            [Path(t[0]).parent.name + "/" + Path(t[0]).stem for t in tones])


def recipe(root: Path = CLEAN, keep: float = KEEP, vowel_mix: float = VOWEL_MIX, scale: float = SCALE,
           breath: float = BREATH, tilt: float = TILT, british: bool = True) -> arpasing.Recipe:
    src, noise, burst, tones = parts(root, keep)
    return arpasing.Recipe("Joki_EN", f0=F0, scale=scale, source=blend(src, young_voice(breath, tilt), vowel_mix),
                           noise=noise, burst=burst, onset_post=pop(root), vowel_mix=0.0,
                           vowels=BRITISH_EN if british else None, non_rhotic=british,
                           readme=f"Joki in English: chill and cozy, a British lad in his late teens (non-rhotic, "
                                  f"British vowels). Steam device startups ({', '.join(tones)}) over a relaxed young "
                                  "voice, Steam Deck swooshes for the hiss, Deck typing / slider ticks and the "
                                  "screenshot click as releases, Steam's soft pop on each syllable.")


def build_japanese(out: Path, sfx_bank: Path | None = None, name: str = "Joki", root: Path = CLEAN) -> Path:
    src, noise, burst, tones = parts(root, KEEP)
    source = blend(src, young_voice(BREATH, TILT), VOWEL_MIX)
    tap = pop(root)

    def post(y, c, m, rng):
        return tap(y, c, rng)
    old = synthbank.F0, synthbank.FORMANT_SCALE
    synthbank.F0, synthbank.FORMANT_SCALE = F0, JP_SCALE
    try:
        with british_japanese(), _hooks(source, noise, burst, post):
            bank = synthbank.build(out, name)
    finally:
        synthbank.F0, synthbank.FORMANT_SCALE = old
    n = _merge_sfx(bank, sfx_bank)
    (bank / "readme.txt").write_text(
        f"{name}\nA male Japanese CV voicebank made of Steam sounds: SteamOS, Steam Deck, Steam Machine, Steam Frame, "
        "Big Picture, the Steam desktop app and the Steam website. Chill and cozy, a British lad in his late teens - "
        "and he sings Japanese with a British accent (British vowel colours, an English r instead of the flap, "
        f"aspirated p/t/k). Steam device startups ({', '.join(tones)}) over a relaxed young voice, Steam Deck swooshes "
        "for the hiss, Deck typing / slider ticks and the screenshot click as k/t/p releases, Steam's soft pop on each "
        f"syllable. Same synthesiser as MADO.\n{n} one-shot Steam sounds as <device>_<name>.\nNo recorded voice: "
        "every clip was screened.\nRender with voice2utau.sing --engine psola.\nThe source sounds belong to Valve. "
        "Personal use only.\n", encoding="utf-8")
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
