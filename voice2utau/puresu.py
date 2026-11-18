"""Puresu: a male voicebank made of PlayStation sounds (PS1 to PS5, PSP, PS2 hardware). Sophisticated, with an Xbox-ish
sleek, dark, techy vibe.

Voice: the boot and menu tones of four generations blended - PSP opening, PS3 cold boot, PS4 startup and the PS5
checkout chime - with a little real voice layered under them so the vowels read, and a faint raw layer of the PS4
startup pad on the vowels after the vocal filter (that pad is where the Xbox-style sheen comes from). Consonant hiss
from PS5/PS4 menu swooshes and the PS2's cooling fan; k/t/p releases from the PS3 XMB cursor, the PS4/PS5 on-screen
keyboard and the PS2 power switch. Signature: the faint XMB cursor tick at the start of every syllable.
Japanese CV bank (synthbank) and English Arpasing bank (arpasing) from the same recipe; every clip voice-screened and
the 101soundboards spoken watermark cut off.

    python -m voice2utau.puresu japanese OUT [--sfx-bank DIR]
    python -m voice2utau.arpasing build puresu OUT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import arpasing, audio, ninchan, synthbank
from .characters import _hooks, _merge_sfx

SR = synthbank.SR
F0 = 123.47                        # B2: a baritone, a step under ANDRO's C3
SCALE = 0.84                       # a longer vocal tract than ANDRO (0.86): deeper, darker
JP_SCALE = 0.86                    # Japanese has five vowels: a too-dark tract pulls them together
KEEP, VOWEL_MIX, DIRECT_DB = 0.8, 0.4, -24.0   # 0.3 / -15 dB: 45% of English word sounds heard; this ~60-65%
TICK_DB = -22.0
CLEAN = Path(__file__).resolve().parent.parent / "voicebanks" / "puresu" / "stage"
TONES = ["psp/PSP_1_00_opening_plugin_rco_snd_opening_ogg", "ps3/Coldboot_Multi", "ps4/Ps4_Startup",
         "ps5/Purchase_Universal_Checkout"]
PAD = ["ps4/Ps4_Startup", "ps4/log_login_se_log_login", "ps4/3dvr_notification_se_Morpheus_Noti"]
HISS = ["ps5/Open_Home", "ps5/Close_Option_Menu", "ps5/Take_Screenshot", "ps4/boot_game_boot_se_boot_game_boo",
        "ps4/Boot_Miniapp", "hw/PS2_Ambient_Cooling_Fan"]
CLICKS = ["ps3/cursor", "ps3/decide", "ps3/cancel", "ps4/osk_key_press_se_osk_key_press", "ps5/Focus_Move_in_Keyboard",
          "ps5/Text_Input", "hw/PS2_Power_Supply_Switch"]
TICK = "ps3/cursor"


def _files(names: list[str], root: Path = CLEAN) -> list[Path]:
    return [p for p in (root / f"{n}.wav" for n in names) if p.exists()]


def tick(root: Path = CLEAN, level_db: float = TICK_DB, ms: float = 30.0):
    """The XMB cursor tick (its first ~30 ms), faint, right at the start of each syllable."""
    f = _files([TICK], root)
    if not f:
        return lambda y, c, rng: (y, c)
    x, sr = audio.read_wav(f[0])
    x = audio.resample(x, sr, SR) if sr != SR else x
    a = int(np.argmax(np.abs(x) > 0.05 * np.abs(x).max()))
    t = np.asarray(x[a:a + int(ms / 1000 * SR)], dtype=np.float64)
    t = t * np.clip((len(t) - np.arange(len(t))) / (0.01 * SR), 0, 1)
    t = t / max(1e-9, float(np.sqrt((t ** 2).mean())))

    def post(y: np.ndarray, c: int, rng) -> tuple[np.ndarray, int]:
        y = np.asarray(y, dtype=np.float64).copy()
        vow = y[c + int(0.06 * SR): c + int(0.2 * SR)]
        ref = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
        s = max(0, c - int(0.004 * SR))
        k = min(len(t), len(y) - s)
        y[s:s + k] += t[:k] * ref * 10 ** (level_db / 20)
        return y, c
    return post


def parts(root: Path = CLEAN, keep: float = KEEP):
    old = synthbank.F0
    synthbank.F0 = F0
    try:
        tones = ninchan.pick_carriers([], top=len(TONES), keep=keep, files=_files(TONES, root))
        pad = ninchan.pick_carriers([], top=len(PAD), keep=1.0, files=_files(PAD, root))
    finally:
        synthbank.F0 = old
    if not tones:
        raise ValueError("no steady PlayStation tone found for Puresu's voice")
    hiss, clicks = _files(HISS, root), _files(CLICKS, root)
    return (ninchan.make_source([t[3] for t in tones]), ninchan.make_noise(hiss, mix=0.55) if hiss else None,
            ninchan.make_burst(clicks) if clicks else None, [Path(t[0]).stem for t in tones],
            ninchan.make_source([t[3] for t in pad]) if pad else None)


def recipe(root: Path = CLEAN, keep: float = KEEP, vowel_mix: float = VOWEL_MIX, direct_db: float | None = DIRECT_DB,
           scale: float = SCALE) -> arpasing.Recipe:
    src, noise, burst, tones, pad = parts(root, keep)
    return arpasing.Recipe("Puresu_EN", f0=F0, scale=scale, source=src, noise=noise, burst=burst, onset_post=tick(root),
                           vowel_mix=vowel_mix, direct=pad if direct_db is not None else None,
                           direct_db=direct_db if direct_db is not None else 0.0,
                           readme=f"Puresu in English: sophisticated, sleek and dark. PlayStation boot tones "
                                  f"({', '.join(tones)}) with a real voice layered under them, a faint PS4-startup pad "
                                  "on the vowels, PS menu swooshes and the PS2 fan for the hiss, XMB cursor / on-screen "
                                  "keyboard clicks as releases, a faint XMB tick on each syllable.")


def build_japanese(out: Path, sfx_bank: Path | None = None, name: str = "Puresu", root: Path = CLEAN) -> Path:
    src, noise, burst, tones, pad = parts(root, KEEP)
    tk = tick(root)
    layered = arpasing.layered(src, VOWEL_MIX)
    g = 10 ** (DIRECT_DB / 20)

    def post(y, c, m, rng):                                  # the raw pad on the vowel, after the vocal filter
        y, c = tk(y, c, rng)
        n = len(y) - c
        if pad is not None and n > int(0.05 * SR):
            vow = y[c + int(0.06 * SR): c + int(0.25 * SR)]
            ref = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
            d = pad(n, rng)
            env = np.clip(np.arange(n) / (0.015 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.04 * SR), 0, 1)
            d = d * env / max(1e-9, float(np.sqrt((d ** 2).mean()))) * ref * g
            y = y.copy()
            y[c:] += d
        return y, c
    old = synthbank.F0, synthbank.FORMANT_SCALE
    synthbank.F0, synthbank.FORMANT_SCALE = F0, JP_SCALE
    try:
        with _hooks(layered, noise, burst, post):
            bank = synthbank.build(out, name)
    finally:
        synthbank.F0, synthbank.FORMANT_SCALE = old
    n = _merge_sfx(bank, sfx_bank)
    (bank / "readme.txt").write_text(
        f"{name}\nA male Japanese CV voicebank made of PlayStation sounds (PS1, PS2, PS3, PS4, PS5, PSP). Sophisticated "
        f"and sleek: PlayStation boot tones ({', '.join(tones)}) with a little real voice under them and a faint "
        "PS4-startup pad on the vowels, PS menu swooshes and the PS2 cooling fan for the hiss, XMB cursor / on-screen "
        "keyboard / PS2 power-switch clicks as k/t/p releases, a faint XMB tick on each syllable. Same synthesiser as "
        f"MADO.\n{n} one-shot PlayStation sounds as <console>_<name>.\nNo recorded voice: every clip was screened "
        "(the PS5 voice-assistant phrases and the soundboard's spoken watermark are out).\nRender with "
        "voice2utau.sing --engine psola.\nThe source sounds belong to Sony Interactive Entertainment. Personal use "
        "only.\n", encoding="utf-8")
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
