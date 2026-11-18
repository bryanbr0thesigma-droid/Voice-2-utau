"""LINX: a male voicebank made of desktop Linux sounds - Ubuntu (the classic theme with its login drums, and Yaru),
GNOME (freedesktop theme, GNOME 2), KDE (Oxygen, Plasma 6 Ocean), Linux Mint, elementary OS, Deepin and ChromeOS.
Friendly open-source; a man in his late twenties to early thirties. Android (ANDRO), Meta Quest (Kaso) and SteamOS are
not in it.

Voice: the login / logout tones of several distros blended (many distros, one guy) over a healthy young man's voice
(more real voice than the console characters, at C#3 with an adult male vocal tract), plus a faint, warm (low-passed) raw
layer of Ubuntu's Yaru login tone on the vowels. Hiss from trash / slide / shade swooshes; k/t/p releases from KDE, GNOME,
ChromeOS, Mint and elementary button clicks. Signature: a soft tap of Ubuntu's original login drums at the start of
every syllable. Every clip voice-screened (the freedesktop speaker-test voices are out).

    python -m voice2utau.linx japanese OUT [--sfx-bank DIR]
    python -m voice2utau.arpasing build linx OUT
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scipy.signal import butter, lfilter, sosfilt

from . import arpasing, audio, ninchan, synthbank
from .characters import _hooks, _merge_sfx

SR = synthbank.SR
F0 = 138.59                        # C#3: a man around thirty, between ANDRO (C3) and UNIX (D3)
SCALE = 0.84                       # an adult male vocal tract, chesty (0.87 + a bright source read like UNIX)
JP_SCALE = SCALE                   # with the breathy, warm voice Japanese reads as well at 0.84 (70/101 vowels)
KEEP, VOWEL_MIX, DIRECT_DB = 0.75, 0.7, -19.0   # mostly real voice: a man around thirty, ~67% of English word sounds heard
DIRECT_LP = 4000.0                 # the raw layer warms the mids only: unfiltered, its top end made him sound like UNIX
JP_DIRECT_DB = -24.0               # the warm layer sits on the formants: -16 dB cost Japanese vowels 71 -> 56/101
BREATH, TILT = 0.2, 0.35           # breathy and warm: the opposite of UNIX's dry, gritty precision
DRUM_DB = -17.0
CLEAN = Path(__file__).resolve().parent.parent / "voicebanks" / "linx" / "stage"
TONES = ["mint/linuxmint-login", "yaru/desktop-login", "kde/Oxygen-Sys-Log-In-Long", "gnome2/shutdown1",
         "deepin/desktop-login", "chromeos/startup"]
DIRECT = ["yaru/desktop-login"]
HISS = ["gnome/trash-empty", "deepin/trash-empty", "mint/window-slide", "kde/Oxygen-Window-Shade-Up",
        "kde/Oxygen-Window-Shade-Down", "chromeos/accessibility_object_delete", "gnome2/card_shuffle"]
CLICKS = ["kde6/button-pressed", "kde6/button-pressed-modifier", "gnome2/clicked", "chromeos/accessibility_touch_type",
          "mint/button-toggle-off", "elementary/audio-volume-change", "deepin/screen-capture", "gnome/device-added"]
DRUM = "ubuntu/desktop-login"


def _files(names: list[str], root: Path = CLEAN) -> list[Path]:
    return [p for p in (root / f"{n}.wav" for n in names) if p.exists()]


def drum_tap(root: Path = CLEAN, level_db: float = DRUM_DB, ms: float = 70.0):
    """The first hit of Ubuntu's old login drums, short and soft, at the start of each syllable."""
    f = _files([DRUM], root)
    if not f:
        return lambda y, c, rng: (y, c)
    x, sr = audio.read_wav(f[0])
    x = np.asarray(audio.resample(x, sr, SR) if sr != SR else x, dtype=np.float64)
    a = int(np.argmax(np.abs(x) > 0.2 * np.abs(x).max()))
    t = x[max(0, a - int(0.002 * SR)): a + int(ms / 1000 * SR)].copy()
    t *= np.exp(-np.arange(len(t)) / (0.025 * SR))
    t /= max(1e-9, float(np.sqrt((t ** 2).mean())))

    def post(y: np.ndarray, c: int, rng) -> tuple[np.ndarray, int]:
        y = np.asarray(y, dtype=np.float64).copy()
        vow = y[c + int(0.06 * SR): c + int(0.2 * SR)]
        ref = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
        s = max(0, c - int(0.006 * SR))
        k = min(len(t), len(y) - s)
        y[s:s + k] += t[:k] * ref * 10 ** (level_db / 20)
        return y, c
    return post


def parts(root: Path = CLEAN, keep: float = KEEP):
    old = synthbank.F0
    synthbank.F0 = F0
    try:
        tones = ninchan.pick_carriers([], top=len(TONES), keep=keep, files=_files(TONES, root))
        raw = ninchan.pick_carriers([], top=len(DIRECT), keep=1.0, files=_files(DIRECT, root))
    finally:
        synthbank.F0 = old
    if not tones:
        raise ValueError("no steady Linux tone found for LINX's voice")
    hiss, clicks = _files(HISS, root), _files(CLICKS, root)
    return (ninchan.make_source([t[3] for t in tones]), ninchan.make_noise(hiss, mix=0.55) if hiss else None,
            ninchan.make_burst(clicks) if clicks else None, [Path(t[0]).parent.name + "/" + Path(t[0]).stem
                                                             for t in tones],
            _warm(ninchan.make_source([t[3] for t in raw])) if raw else None)


def _warm(source, fc: float = DIRECT_LP):
    sos = butter(4, fc, "low", fs=SR, output="sos")
    return lambda n, rng: sosfilt(sos, source(n, rng))


def young_voice(breath: float = BREATH, tilt: float = TILT, jitter: float = 0.005, shimmer: float = 0.035):
    """A relaxed young man's voice source: glottal pulses with a gentle tilt (warm, not bright) and breath riding on
    each cycle - soft and friendly, the opposite of UNIX's dry, gritty precision."""
    hp = butter(2, 1200.0, "high", fs=SR, output="sos")
    sm = butter(1, 400.0, "low", fs=SR, output="sos")

    def source(n: int, rng: np.random.Generator) -> np.ndarray:
        g = synthbank.glottal(n, rng, jitter=jitter, shimmer=shimmer)
        g = lfilter([1 - tilt], [1, -tilt], g)
        noise = sosfilt(hp, rng.standard_normal(n)) * np.maximum(sosfilt(sm, np.abs(g)), 0)
        g /= max(1e-9, float(np.sqrt((g ** 2).mean())))
        noise /= max(1e-9, float(np.sqrt((noise ** 2).mean())))
        y = np.sqrt(1 - breath ** 2) * g + breath * noise
        return y / max(1e-9, np.abs(y).max())
    return source


def blend(tones, voice, mix: float):
    """The distro tones with the young voice under them (`mix` = the voice's share at equal loudness)."""
    def source(n: int, rng: np.random.Generator) -> np.ndarray:
        a, b = tones(n, rng), voice(n, rng)
        a = a / max(1e-9, float(np.sqrt((a ** 2).mean())))
        b = b / max(1e-9, float(np.sqrt((b ** 2).mean())))
        y = a * (1 - mix) + b * mix
        return y / max(1e-9, np.abs(y).max())
    return source


def recipe(root: Path = CLEAN, keep: float = KEEP, vowel_mix: float = VOWEL_MIX, direct_db: float | None = DIRECT_DB,
           scale: float = SCALE, breath: float = BREATH, tilt: float = TILT) -> arpasing.Recipe:
    src, noise, burst, tones, raw = parts(root, keep)
    return arpasing.Recipe("LINX_EN", f0=F0, scale=scale, source=blend(src, young_voice(breath, tilt), vowel_mix),
                           noise=noise, burst=burst, onset_post=drum_tap(root), vowel_mix=0.0,
                           direct=raw if direct_db is not None else None,
                           direct_db=direct_db if direct_db is not None else 0.0,
                           readme=f"LINX in English: friendly open-source, a man around thirty. Desktop Linux login "
                                  f"tones ({', '.join(tones)}) over a real young man's voice, a faint Yaru login layer "
                                  "on the vowels, Linux trash / slide swooshes for the hiss, KDE / GNOME / ChromeOS "
                                  "clicks as releases, a soft tap of Ubuntu's login drums on each syllable.")


def build_japanese(out: Path, sfx_bank: Path | None = None, name: str = "LINX", root: Path = CLEAN) -> Path:
    src, noise, burst, tones, raw = parts(root, KEEP)
    layered = blend(src, young_voice(), VOWEL_MIX)
    g = 10 ** (JP_DIRECT_DB / 20)
    tap = drum_tap(root)

    def post(y, c, m, rng):                                  # drum tap, then the raw Yaru layer after the filter
        y, c = tap(y, c, rng)
        n = len(y) - c
        if raw is not None and n > int(0.05 * SR):
            vow = y[c + int(0.06 * SR): c + int(0.25 * SR)]
            ref = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
            d = raw(n, rng)
            env = np.clip(np.arange(n) / (0.015 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.04 * SR), 0, 1)
            y = y.copy()
            y[c:] += d * env / max(1e-9, float(np.sqrt((d ** 2).mean()))) * ref * g
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
        f"{name}\nA male Japanese CV voicebank made of desktop Linux sounds: Ubuntu, GNOME, KDE, Linux Mint, "
        "elementary OS, Deepin and ChromeOS. Friendly open-source, a man in his late twenties to early thirties: "
        f"login tones ({', '.join(tones)}) over a real young man's voice, a faint Yaru login layer on the vowels, "
        "Linux swooshes for the hiss, KDE / GNOME / ChromeOS clicks as k/t/p releases, a soft tap of Ubuntu's login "
        f"drums on each syllable. Same synthesiser as MADO.\n{n} one-shot Linux sounds as <distro>_<name>.\n"
        "No recorded voice: every clip was screened (the freedesktop speaker-test voices are out).\nNo Android, "
        "Meta Quest or SteamOS.\nRender with voice2utau.sing --engine psola.\nThe sounds come from their "
        "projects' sound themes (mostly free licences, see each project); personal use.\n", encoding="utf-8")
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
