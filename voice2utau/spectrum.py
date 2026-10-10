"""Match the long-term spectrum of a set of wavs to a reference (and optionally pull it toward a brighter one).

RVC models trained briefly on a small dataset can come out dull. `match_dir` measures the average spectrum of a
reference set (e.g. the character's real recordings), compares it with the converted clips and applies the
difference as a smooth zero-phase EQ to every clip (each clip keeps its own level). `toward` blends the target
halfway to a second set (e.g. a clearer voice model's output); that set's narrow peaks (> +12 dB over the reference)
are not copied.

    python -m voice2utau.spectrum SRC_DIR OUT_DIR --reference WAV [WAV ...] [--toward WAV [WAV ...]]
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.signal import stft

from . import audio

SR = audio.SR_BANK
NF = 2048


def ltas(files: list[Path]) -> np.ndarray:
    """Mean normalised power spectrum over the loudest 40 % of frames of each file."""
    acc, n = np.zeros(NF // 2 + 1), 0
    for p in files:
        x, sr = audio.read_wav(p)
        x = audio.resample(x, sr, SR) if sr != SR else x
        if len(x) < 4096:
            continue
        S = np.abs(stft(x, SR, nperseg=NF, noverlap=NF - 512, boundary=None)[2]) ** 2
        e = S.sum(0)
        m = S[:, e > np.percentile(e, 60)].mean(1)
        acc += m / m.sum()
        n += 1
    if not n:
        raise ValueError("no usable wav files")
    return acc / n


def smooth_db(pw: np.ndarray, sigma_oct: float = 0.25) -> np.ndarray:
    f = np.fft.rfftfreq(NF, 1 / SR)
    lf = np.log2(np.maximum(f, 20))
    out = np.empty_like(pw)
    for i in range(len(f)):
        w = np.exp(-0.5 * ((lf - lf[i]) / sigma_oct) ** 2)
        out[i] = (pw * w).sum() / w.sum()
    return 10 * np.log10(out + 1e-12)


def gain_db(reference: list[Path], source: list[Path], toward: list[Path] | None = None,
            lo: float = -10.0, hi: float = 14.0) -> tuple[np.ndarray, np.ndarray]:
    f = np.fft.rfftfreq(NF, 1 / SR)
    target = smooth_db(ltas(reference))
    if toward:
        t = smooth_db(ltas(toward))
        target = 0.5 * target + 0.5 * np.minimum(t, target + 12.0)
    g = np.clip(target - smooth_db(ltas(source)), lo, hi)
    g[f < 80] = 0.0
    hf = f > 9000
    g[hf] *= np.clip((12000 - f[hf]) / 3000, 0, 1)                     # leave the extremes alone
    return f, g


def read_pre_ms(oto: Path) -> dict[str, float]:
    """wav file name -> preutterance (ms) from an oto.ini (first entry per file)."""
    out: dict[str, float] = {}
    if not oto.exists():
        return out
    for line in oto.read_bytes().decode("cp932", "replace").splitlines():
        if "=" in line:
            wav, rest = line.split("=", 1)
            f = rest.split(",")
            if len(f) >= 6 and wav not in out:
                out[wav] = float(f[4])             # alias, offset, consonant, cutoff, preutterance, overlap
    return out


def _eq(x: np.ndarray, sr: int, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    N = 1 << int(np.ceil(np.log2(len(x) * 2)))
    gain = 10 ** (np.interp(np.fft.rfftfreq(N, 1 / sr), f, g) / 20)
    return np.fft.irfft(np.fft.rfft(x, N) * gain, N)[:len(x)].astype("float32")


def apply_eq(src: Path, dst: Path, f: np.ndarray, g: np.ndarray, cons_cap_db: float | None = 2.0) -> int:
    """EQ every wav in `src` into `dst` (other files are copied).

    The EQ is measured on voiced sound, so boosting the consonant part as well makes fricatives (zh ch sh c ...) harsher
    and louder against their vowels. With `cons_cap_db` (and an oto.ini in `src`) the part before the preutterance gets the
    gain capped at that many dB, crossfading into the full EQ over 30 ms at the vowel start. The vowel part keeps the level
    it would have had with the plain EQ."""
    dst.mkdir(parents=True, exist_ok=True)
    pre_of = read_pre_ms(src / "oto.ini") if cons_cap_db is not None else {}
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() != ".wav":
            shutil.copy2(p, dst / p.name)
            continue
        x, sr = audio.read_wav(p)
        y = _eq(x, sr, f, g)
        y *= np.sqrt((x ** 2).mean() / max(1e-12, (y ** 2).mean()))
        if p.name in pre_of:
            c = int(pre_of[p.name] / 1000 * sr)
            w = int(0.015 * sr)
            yc = _eq(x, sr, f, np.minimum(g, cons_cap_db))
            v = slice(c + w, c + w + int(0.1 * sr))                      # vowel window: same level as the plain EQ
            if len(y[v]) > 200:
                yc *= np.sqrt((y[v] ** 2).mean() / max(1e-12, (yc[v] ** 2).mean()))
            mix = np.clip((np.arange(len(x)) - (c - w)) / (2 * w), 0, 1).astype("float32")
            y = yc * (1 - mix) + y * mix
        audio.write_wav(dst / p.name, np.clip(y, -0.99, 0.99), sr)
        n += 1
    return n


def match_dir(src: Path, dst: Path, reference: list[Path], toward: list[Path] | None = None,
              cons_cap_db: float | None = 2.0) -> np.ndarray:
    f, g = gain_db(reference, sorted(src.glob("*.wav")), toward)
    apply_eq(src, dst, f, g, cons_cap_db)
    return np.interp([200, 500, 1000, 2000, 3000, 5000, 8000], f, g)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--reference", nargs="+", type=Path, required=True)
    ap.add_argument("--toward", nargs="+", type=Path)
    a = ap.parse_args()
    g = match_dir(a.src, a.out, a.reference, a.toward)
    print("gain dB at 200/500/1k/2k/3k/5k/8k Hz:", " ".join(f"{v:+.1f}" for v in g))


if __name__ == "__main__":
    main()
