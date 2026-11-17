"""nin-Chan CV: a Japanese CV voicebank whose voice is made of Nintendo system sounds.

Same synthesiser as MADO (`synthbank`), with the raw material swapped:
  * voice  - one pitch period cut from the steadiest part of tonal system sounds (averaged over several periods),
             partly spectrally flattened so the vowel formants can be imposed while the timbre stays, then looped at F0
  * noise  - grains of noisy system sounds (shutter, whooshes, clicks), flattened to a broadband bed and mixed 50/50
             with white noise, used for aspiration, fricatives and stop bursts
The voice period gets MADO's voice-source spectral tilt and keeps 30% of its own harmonic ripple plus its phases:
more ripple = more Nintendo, less intelligible (phoneme recogniser, both C and V right: 20/101 at 30%, 9/101 at 100%).

    python -m voice2utau.ninchan SOUND_DIR [SOUND_DIR ...] -o OUT --name nin-Chan
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from . import audio, synthbank

SR = synthbank.SR
N = 2048                                   # wavetable length (one period)


def _pitch(x: np.ndarray, sr: int) -> np.ndarray:
    import parselmouth
    p = parselmouth.Sound(x.astype("float64"), sr).to_pitch(0.01, 80, 1200)
    return p.selected_array["frequency"]


def steady_window(x: np.ndarray, sr: int, win_s: float = 0.08) -> tuple[int, float, float] | None:
    """(start sample, f0, wobble cents) of the most stable fully voiced window."""
    f = _pitch(x, sr)
    w = max(3, int(win_s / 0.01))
    best = None
    for i in range(0, len(f) - w):
        seg = f[i:i + w]
        if (seg <= 0).any():
            continue
        f0 = float(np.median(seg))
        wob = float(np.std(1200 * np.log2(seg / f0)))
        if best is None or wob < best[2]:
            best = (int(i * 0.01 * sr), f0, wob)
    return best


def wavetable(x: np.ndarray, sr: int, start: int, f0: float, periods: int = 8) -> np.ndarray:
    """Average `periods` consecutive periods (each resampled to N points, circularly aligned to the first)."""
    T = sr / f0
    ref, acc, used = None, np.zeros(N), 0
    for k in range(periods):
        a = start + k * T
        if a + T + 2 >= len(x):
            break
        idx = a + np.arange(N) * T / N
        per = np.interp(idx, np.arange(len(x)), x)
        if ref is None:
            ref = per
        else:
            sh = int(np.argmax(np.fft.irfft(np.fft.rfft(ref) * np.conj(np.fft.rfft(per)))))
            per = np.roll(per, sh)
        acc += per
        used += 1
    wt = acc / max(1, used)
    wt -= wt.mean()
    return wt / max(1e-9, np.abs(wt).max())


def richness(wt: np.ndarray, f0: float = synthbank.F0) -> int:
    """Harmonics below 5 kHz within 30 dB of the strongest, if this period is played at f0."""
    m = np.abs(np.fft.rfft(wt))[1:int(5000 / f0) + 1]
    db = 20 * np.log10(m / max(1e-12, m.max()) + 1e-12)
    return int((db > -30).sum())


def _glottal_envelope() -> np.ndarray:
    """Harmonic magnitudes of MADO's own voice-source period (differentiated Rosenberg pulse + gentle tilt)."""
    tp, te = int(0.48 * N), int(0.6 * N)
    p = np.zeros(N)
    p[:tp] = 0.5 * (1 - np.cos(np.pi * np.arange(tp) / tp))
    p[tp:te] = np.cos(0.5 * np.pi * np.arange(te - tp) / max(1, te - tp))
    d = np.diff(p, prepend=0.0)
    G = np.abs(np.fft.rfft(d))
    k = np.arange(len(G))
    tilt = np.abs(0.6 / (1 - 0.4 * np.exp(-2j * np.pi * k * synthbank.F0 / SR)))
    return G * tilt / max(1e-12, (G * tilt)[1:].max())


def flatten(wt: np.ndarray, keep: float = 0.3, smooth: int = 5) -> np.ndarray:
    """Give the period a voice-like spectral tilt while keeping its own phases and part of its harmonic pattern.

    magnitude_k = glottal_envelope_k * (own_magnitude_k / own_smoothed_envelope_k) ** keep ;  phase_k = own phase_k
    keep = 0 -> a plain voice source with Nintendo phases; keep = 1 -> the full Nintendo harmonic ripple on a voice tilt."""
    X = np.fft.rfft(wt)
    mag = np.abs(X) + 1e-9
    env = np.exp(np.convolve(np.log(mag), np.ones(smooth) / smooth, "same"))
    G = _glottal_envelope()
    Y = G * (mag / env) ** keep * np.exp(1j * np.angle(X))
    Y[0] = 0
    Y[int(SR / 2 / synthbank.F0) + 1:] = 0                 # nothing above Nyquist once played at F0
    y = np.fft.irfft(Y, N)
    return y / max(1e-9, np.abs(y).max())


def pick_carriers(dirs: list[Path], top: int = 3, max_wobble: float = 25.0) -> list[tuple[str, float, int, np.ndarray]]:
    """[(file, f0, richness, flattened wavetable)] for the richest steady tonal sounds."""
    out = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.wav")):
            x, sr = audio.read_wav(p)
            if len(x) < sr * 0.12:
                continue
            sw = steady_window(x, sr)
            if sw is None or sw[2] > max_wobble:
                continue
            wt = wavetable(x, sr, sw[0], sw[1])
            out.append((str(p), sw[1], richness(wt), wt))
    out.sort(key=lambda r: -r[2])
    return [(f, f0, r, flatten(wt)) for f, f0, r, wt in out[:top]]


def make_source(tables: list[np.ndarray], weights: list[float] | None = None):
    """SOURCE_FN for synthbank: loop the (blended) wavetable at F0 with slight jitter and shimmer."""
    w = np.array(weights or [1.0] * len(tables), dtype=float)
    wt = sum(t * x for t, x in zip(tables, w / w.sum()))
    wt = wt / max(1e-9, np.abs(wt).max())

    def source(n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros(n)
        i = 0
        while i < n:
            T = max(8, int(round(SR / (synthbank.F0 * (1 + 0.004 * rng.standard_normal())))))
            per = np.interp(np.arange(T) * N / T, np.arange(N), wt) * (1 + 0.03 * rng.standard_normal())
            m = min(T, n - i)
            out[i:i + m] = per[:m]
            i += T
        return out / max(1e-9, np.abs(out).max())
    return source


def make_noise(files: list[Path], seconds: float = 12.0, seed: int = 0, mix: float = 0.5):
    """NOISE_FN for synthbank: a flattened bed of 25 ms grains from noisy system sounds."""
    rng = np.random.default_rng(seed)
    clips = []
    for p in files:
        x, sr = audio.read_wav(p)
        x = audio.resample(x, sr, SR) if sr != SR else x
        if len(x) > int(0.03 * SR):
            clips.append(x / max(1e-9, np.abs(x).max()))
    g = int(0.025 * SR)
    win = np.hanning(2 * g)
    bed = np.zeros(int(seconds * SR) + 2 * g)
    for pos in range(0, len(bed) - 2 * g, g):
        c = clips[rng.integers(len(clips))]
        a = rng.integers(0, max(1, len(c) - 2 * g))
        grain = c[a:a + 2 * g]
        bed[pos:pos + len(grain)] += grain * win[:len(grain)] * (1 if rng.random() < 0.5 else -1)
    S = np.fft.rfft(bed)
    mag = np.abs(S) + 1e-9
    k = np.ones(201) / 201
    S = S / np.exp(np.convolve(np.log(mag), k, "same"))      # flatten to broadband, keep the grain texture
    bed = np.fft.irfft(S, len(bed))
    bed = bed / np.std(bed)

    def noise(n: int, rng2: np.random.Generator) -> np.ndarray:
        if n >= len(bed):
            part = np.tile(bed, n // len(bed) + 1)[:n]
        else:
            a = int(rng2.integers(0, len(bed) - n))
            part = bed[a:a + n]
        # half Nintendo grains, half white noise: measured 39 vs 42 consonants right (white) vs 29 (grains only)
        return mix * part + (1 - mix) * rng2.standard_normal(n)
    return noise


def noisy_files(dirs: list[Path], max_hnr: float = 3.0, limit: int = 12) -> list[Path]:
    import parselmouth
    rows = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.wav")):
            x, sr = audio.read_wav(p)
            if len(x) < int(0.03 * sr):
                continue
            h = np.array(parselmouth.Sound(x.astype("float64"), sr).to_harmonicity_cc(0.01, 80).values[0])
            h = h[h > -100]
            rows.append((float(np.median(h)) if len(h) else -10.0, p))
    rows.sort(key=lambda r: r[0])
    return [p for hnr, p in rows if hnr <= max_hnr][:limit]


@contextlib.contextmanager
def nintendo_voice(dirs: list[Path], top: int = 3):
    carriers = pick_carriers(dirs, top)
    if not carriers:
        raise ValueError("no steady tonal sound found to use as the voice")
    noisy = noisy_files(dirs)
    old = synthbank.SOURCE_FN, synthbank.NOISE_FN
    synthbank.SOURCE_FN = make_source([c[3] for c in carriers], [c[2] for c in carriers])
    if noisy:
        synthbank.NOISE_FN = make_noise(noisy)
    try:
        yield {"carriers": [(Path(f).name, round(f0, 1), r) for f, f0, r, _ in carriers],
               "noise": [Path(p).name for p in noisy]}
    finally:
        synthbank.SOURCE_FN, synthbank.NOISE_FN = old


def build(dirs: list[Path], out_dir: Path, name: str = "nin-Chan", top: int = 3) -> tuple[Path, dict]:
    with nintendo_voice(dirs, top) as info:
        root = synthbank.build(out_dir, name)
    (root / "readme.txt").write_text(
        f"{name}\nA Japanese CV voicebank whose voice is made of Nintendo system sounds: the vowels are a pitch period cut "
        f"from {', '.join(c[0] for c in info['carriers'])} (looped at C4, shaped into vowels), the consonant noise is made "
        f"from {len(info['noise'])} noisy system sounds. Same synthesiser as MADO (voice2utau.synthbank).\n"
        "Aliases: hiragana (か) and romaji (ka); ん is also 'n'; 息 / br is a breath.\n"
        "The source sounds belong to Nintendo. Personal use only - do not redistribute.\n", encoding="utf-8")
    return root, info


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--name", default="nin-Chan")
    ap.add_argument("--top", type=int, default=3, help="how many tonal sounds to blend into the voice")
    a = ap.parse_args()
    root, info = build(a.dirs, a.out, a.name, a.top)
    print(f"wrote {root}\nvoice from: {info['carriers']}\nnoise from: {info['noise']}")


if __name__ == "__main__":
    main()
