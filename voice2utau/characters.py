"""Character voices built on MADO's synthesiser, where the sound carries the character instead of a texture on top.

nin-Chan (8-bit lineage): pulse-wave vowels (Game Boy / NES duty-cycle tone, perfectly steady like a chip), NES-style
    LFSR noise for fricatives, real Switch/Wii clicks as k/t/p releases, a tiny menu blip on every vowel onset, light
    8-bit quantisation.
BAKO (sleek, ambient): a glassy pad made from an Xbox sound's own cycle with a three-voice chorus, a slow swell into
    each vowel, the dashboard's air whoosh arriving before every syllable, and a low hum an octave under the note.

    python -m voice2utau.characters ninchan OUT --clicks a.wav b.wav --noise-dirs DIR ... [--sfx-bank DIR]
    python -m voice2utau.characters bako OUT --carrier startup.wav --noise-dirs DIR ...  [--sfx-bank DIR]
"""
from __future__ import annotations

import contextlib
import shutil
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from . import audio, ninchan, synthbank

SR = synthbank.SR
F0 = synthbank.F0


# ----------------------------------------------------------------------------------------------- nin-Chan: 8-bit
def pulse_table(duty: float = 0.25, n: int = ninchan.N) -> np.ndarray:
    """One band-limited period of a pulse wave (harmonics up to Nyquist at F0)."""
    k = np.arange(1, int(SR / 2 / F0) + 1)
    amp = (2 / (np.pi * k)) * np.sin(np.pi * k * duty)
    t = np.arange(n) / n
    y = (amp[:, None] * np.cos(2 * np.pi * k[:, None] * t[None, :])).sum(0)
    return y / np.abs(y).max()


def chip_source(duty: float = 0.25):
    wt = pulse_table(duty)

    def source(n: int, rng: np.random.Generator) -> np.ndarray:
        T = SR / F0                                           # a chip is perfectly steady: no jitter, no shimmer
        ph = (np.arange(n) / T) % 1.0
        y = np.interp(ph * ninchan.N, np.arange(ninchan.N), wt)
        return lfilter([0.7], [1, -0.3], y)                   # take the very top edge off
    return source


def lfsr_noise(clock_hz: float = 22050.0, seed: int = 1):
    """NES-style noise channel: a 15-bit LFSR (feedback bit0 ^ bit1), 1-bit output, sample-and-hold at clock_hz."""
    def noise(n: int, rng: np.random.Generator) -> np.ndarray:
        steps = int(n * clock_hz / SR) + 2
        reg = int(rng.integers(1, 0x7FFF)) or seed
        bits = np.empty(steps)
        for i in range(steps):
            fb = (reg ^ (reg >> 1)) & 1
            reg = (reg >> 1) | (fb << 14)
            bits[i] = 1.0 if reg & 1 else -1.0
        idx = (np.arange(n) * clock_hz / SR).astype(int)
        return bits[idx]
    return noise


def ninchan_post(blip_db: float = -14.0, bits: int = 8):
    def post(y: np.ndarray, c: int, m, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        y = y / max(1e-9, np.abs(y).max()) * 0.8
        n = int(0.025 * SR)                                   # menu blip: a 25 ms square an octave up, quick decay
        t = np.arange(n) / SR
        blip = np.sign(np.sin(2 * np.pi * 2 * F0 * t)) * np.exp(-t / 0.008) * 10 ** (blip_db / 20)
        a = min(max(0, c), len(y) - n)
        y = y.copy()
        y[a:a + n] += blip
        q = 2 ** (bits - 1)                                   # light 8-bit crunch
        y = np.round(np.clip(y, -1, 1) * q) / q
        return y, c
    return post


# ----------------------------------------------------------------------------------------------- BAKO: ambient
def pad_source(table: np.ndarray, detune_cents: float = 6.0):
    """Three slightly detuned copies of one cycle: a soft glassy chorus pad."""
    rates = [2 ** (d / 1200) for d in (-detune_cents, 0.0, detune_cents)]

    def source(n: int, rng: np.random.Generator) -> np.ndarray:
        y = np.zeros(n)
        for r, g in zip(rates, (0.8, 1.0, 0.8)):
            ph = (np.arange(n) * F0 * r / SR + rng.random()) % 1.0
            y += g * np.interp(ph * ninchan.N, np.arange(ninchan.N), table)
        y = lfilter([0.5], [1, -0.5], y)                      # softer top than a voice: glassy, not buzzy
        return y / max(1e-9, np.abs(y).max())
    return source


def bako_post(whoosh_noise, whoosh_ms: float = 170.0, whoosh_db: float = -16.0, hum_db: float = -20.0,
              swell_ms: float = 150.0):
    def post(y: np.ndarray, c: int, m, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        y = y / max(1e-9, np.abs(y).max()) * 0.8
        n = len(y)
        # slow swell into the vowel (soft, never silent)
        s = int(swell_ms / 1000 * SR)
        env = np.ones(n)
        u = np.clip((np.arange(n) - c) / max(1, s), 0, 1)
        env[c:] = 0.6 + 0.4 * (3 * u[c:] ** 2 - 2 * u[c:] ** 3)
        y = y * env
        # low hum an octave under the note, under the vowel only
        vow = y[c:c + int(0.2 * SR)]
        vr = float(np.sqrt((vow ** 2).mean())) if len(vow) else 0.1
        t = np.arange(n) / SR
        hum = np.sin(2 * np.pi * (F0 / 2) * t) * vr * 10 ** (hum_db / 20) * np.sqrt(2)
        hum *= np.clip((np.arange(n) - c) / (0.08 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.05 * SR), 0, 1)
        y = y + hum
        # air whoosh arriving before the syllable: rising band-passed console noise
        w = int(whoosh_ms / 1000 * SR)
        x = whoosh_noise(w, rng)
        out = np.zeros(w)
        blk = 256
        zi = None
        for b0 in range(0, w, blk):                           # 300 Hz -> 3.6 kHz sweep, filter state carried across blocks
            fc = 300 * (12 ** (b0 / w))
            sos = butter(2, [fc * 0.6, min(fc * 1.6, SR / 2 - 100)], "band", fs=SR, output="sos")
            if zi is None:
                zi = np.zeros((sos.shape[0], 2))
            out[b0:b0 + blk], zi = sosfilt(sos, x[b0:b0 + blk], zi=zi)
        out *= np.linspace(0, 1, w) ** 1.5
        out = out / max(1e-9, np.sqrt((out ** 2).mean())) * vr * 10 ** (whoosh_db / 20)
        lead = int(0.6 * w)                                   # the whoosh overlaps the start of the syllable
        y2 = np.zeros(n + w - lead)
        y2[w - lead:] += y
        y2[:w] += out
        return y2, c + (w - lead)
    return post


# ----------------------------------------------------------------------------------------------- building
@contextlib.contextmanager
def _hooks(source, noise, burst, post):
    old = synthbank.SOURCE_FN, synthbank.NOISE_FN, synthbank.BURST_FN, synthbank.POST_FN
    synthbank.SOURCE_FN, synthbank.NOISE_FN, synthbank.BURST_FN, synthbank.POST_FN = source, noise, burst, post
    try:
        yield
    finally:
        synthbank.SOURCE_FN, synthbank.NOISE_FN, synthbank.BURST_FN, synthbank.POST_FN = old


def _merge_sfx(root: Path, sfx: Path | None) -> int:
    if not sfx or not (sfx / "oto.ini").exists():
        return 0
    a = [l for l in (root / "oto.ini").read_bytes().decode("cp932").splitlines() if "=" in l]
    b = [l for l in (sfx / "oto.ini").read_bytes().decode("cp932").splitlines() if "=" in l]
    have = {l.split("=", 1)[1].split(",")[0] for l in a}
    b = [l for l in b if l.split("=", 1)[1].split(",")[0] not in have]
    for p in sfx.iterdir():
        if p.suffix.lower() in (".wav", ".frq") and not (root / p.name).exists():
            shutil.copy2(p, root / p.name)
    (root / "oto.ini").write_bytes(("\n".join(a + b) + "\n").encode("cp932"))
    return len(b)


def build_ninchan(out: Path, clicks: list[Path], sfx_bank: Path | None = None, name: str = "nin-Chan") -> Path:
    with _hooks(chip_source(), lfsr_noise(), ninchan.make_burst(clicks) if clicks else None, ninchan_post()):
        root = synthbank.build(out, name)
    n = _merge_sfx(root, sfx_bank)
    (root / "readme.txt").write_text(
        f"{name}\nAn 8-bit Japanese CV voicebank: pulse-wave vowels (Game Boy / NES duty-cycle tone), NES-style LFSR noise "
        "for fricatives, real Nintendo system clicks as k/t/p releases, a menu blip on each vowel onset, light 8-bit "
        f"quantisation. Same synthesiser as MADO.\nAliases: hiragana and romaji; 息 / br breath; {n} one-shot Nintendo "
        "sounds as <console>_<name>.\nThe click and one-shot sounds belong to Nintendo. Personal use only.\n"
        "Sing it with voice2utau.sing --engine psola to keep the 8-bit detail (WORLD smooths it away).\n", encoding="utf-8")
    return root


def build_bako(out: Path, carrier: Path, noise_dirs: list[Path], sfx_bank: Path | None = None, name: str = "BAKO",
               keep: float = 0.5) -> Path:
    c = ninchan.pick_carriers([], top=1, keep=keep, files=[carrier])
    if not c:
        raise ValueError(f"no steady tone found in {carrier}")
    noisy = ninchan.noisy_files(noise_dirs)
    whoosh = ninchan.make_noise(noisy, mix=0.9) if noisy else None
    noise = ninchan.make_noise(noisy, mix=0.7) if noisy else None
    with _hooks(pad_source(c[0][3]), noise, None, bako_post(whoosh or (lambda n, r: r.standard_normal(n)))):
        root = synthbank.build(out, name)
    n = _merge_sfx(root, sfx_bank)
    (root / "readme.txt").write_text(
        f"{name}\nAn ambient Japanese CV voicebank: glassy chorus-pad vowels cut from {carrier.name}, a slow swell into "
        "each vowel, the Xbox dashboard's air whoosh arriving before each syllable (included in the preutterance), "
        f"and a low hum an octave under the note. Same synthesiser as MADO.\nAliases: hiragana and romaji; 息 / br; "
        f"{n} one-shot Xbox sounds.\nThe source sounds belong to Microsoft. Personal use only.\n"
        "Sing it with voice2utau.sing --engine psola to keep the pad and hum detail.\n", encoding="utf-8")
    return root


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("character", choices=["ninchan", "bako"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--clicks", type=Path, nargs="*", default=[])
    ap.add_argument("--carrier", type=Path)
    ap.add_argument("--noise-dirs", type=Path, nargs="*", default=[])
    ap.add_argument("--sfx-bank", type=Path)
    ap.add_argument("--name")
    a = ap.parse_args()
    if a.character == "ninchan":
        root = build_ninchan(a.out, a.clicks, a.sfx_bank, a.name or "nin-Chan")
    else:
        if not a.carrier:
            ap.error("bako needs --carrier")
        root = build_bako(a.out, a.carrier, a.noise_dirs, a.sfx_bank, a.name or "BAKO")
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
