"""Kaso's Meta Quest space, applied to a rendered vocal: liminal and dreamy.

startup_ir  a reverb made from the Quest startup sound: its swell-and-fade shape and its broad colour, with the
            phases scattered and the spectrum smoothed so it does not ring in the startup chord's key.
halo        her voice an octave up (WORLD: same formants, doubled pitch), like the Quest 'success' chime leaping up
            an octave. Fed only into the reverb, so it hangs above her as a shimmer, not a second singer.
dream       vocal + reverb(vocal + halo).

    python -m voice2utau.quest VOCAL.wav OUT.wav --startup startup.wav [--wet -6] [--halo -3]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((np.asarray(x, dtype=np.float64) ** 2).mean()) + 1e-12)


def _smooth_bins(mag: np.ndarray, freqs: np.ndarray, octaves: float) -> np.ndarray:
    """Average each column over a band `octaves` wide around every bin (log-frequency smoothing)."""
    c = np.cumsum(np.vstack([np.zeros((1, mag.shape[1])), mag]), axis=0)
    f = np.maximum(freqs, 1.0)
    lo = np.searchsorted(freqs, f * 2 ** (-octaves / 2))
    hi = np.maximum(np.searchsorted(freqs, f * 2 ** (octaves / 2)), lo + 1)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def startup_ir(x: np.ndarray, sr: int, length: float = 3.2, smooth_oct: float = 0.5, detail: float = 0.15,
               predelay: float = 0.025, seed: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)[:int(length * sr)]
    nfft, hop = 2048, 512
    win = np.hanning(nfft)
    frames = [x[i:i + nfft] * win for i in range(0, max(1, len(x) - nfft), hop)]
    mag = np.abs(np.fft.rfft(np.array(frames), axis=1)).T                  # (bins, frames)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    mag = (1 - detail) * _smooth_bins(mag, freqs, smooth_oct) + detail * mag
    rng = np.random.default_rng(seed)
    spec = mag * np.exp(2j * np.pi * rng.random(mag.shape))
    ir = np.zeros(len(frames) * hop + nfft)
    for k in range(spec.shape[1]):
        ir[k * hop:k * hop + nfft] += np.fft.irfft(spec[:, k], nfft) * win
    ir = sosfilt(butter(2, 120.0, "high", fs=sr, output="sos"), ir)
    ir = np.concatenate([np.zeros(int(predelay * sr)), ir])
    return (ir / max(1e-12, np.sqrt((ir ** 2).sum()))).astype(np.float32)


def halo(x: np.ndarray, sr: int, ratio: float = 2.0) -> np.ndarray:
    """The vocal at `ratio` times the pitch with the same spectral envelope (WORLD), top-weighted."""
    import pyworld as pw
    x = np.asarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, sr, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    y = pw.synthesize(f0 * ratio, sp, ap, sr, frame_period=5.0)[:len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    return sosfilt(butter(2, 1200.0, "high", fs=sr, output="sos"), y).astype(np.float32)


def dream(x: np.ndarray, sr: int, ir: np.ndarray, wet_db: float = -6.0, halo_db: float = -3.0,
          halo_sig: np.ndarray | None = None) -> np.ndarray:
    """Dry vocal plus the startup reverb of (vocal + octave halo). Output is longer by the reverb tail."""
    x = np.asarray(x, dtype=np.float64)
    send = x.copy()
    if halo_sig is not None:
        send = send + np.asarray(halo_sig, dtype=np.float64)[:len(x)] * (_rms(x) / _rms(halo_sig)) * 10 ** (halo_db / 20)
    wet = fftconvolve(send, ir)
    wet *= _rms(x) / _rms(wet[:len(x)]) * 10 ** (wet_db / 20)
    out = wet
    out[:len(x)] += x
    return out.astype(np.float32)


def main() -> None:
    import argparse
    import soundfile as sf
    from . import audio
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vocal", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--startup", type=Path, required=True, help="the Quest startup sound (the reverb is made from it)")
    ap.add_argument("--wet", type=float, default=-6.0, help="reverb level vs the dry vocal, dB")
    ap.add_argument("--halo", type=float, default=-3.0, help="octave halo into the reverb vs the vocal, dB")
    a = ap.parse_args()
    x, sr = sf.read(a.vocal, dtype="float32")
    s, ssr = audio.read_wav(a.startup)
    s = audio.resample(s, ssr, sr) if ssr != sr else s
    y = dream(x, sr, startup_ir(s, sr), a.wet, a.halo, halo(x, sr))
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * 0.99 / peak
    sf.write(a.out, y, sr, subtype="PCM_16")
    print(f"wrote {a.out}: startup reverb {a.wet:+g} dB, octave halo {a.halo:+g} dB into it")


if __name__ == "__main__":
    main()
