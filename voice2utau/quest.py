"""Kaso's Meta Quest space, applied to a rendered vocal: liminal and dreamy.

startup_ir  a reverb shaped by the Quest startup sound: its swell and fade, a trace of its colour, on an airy
            full-range spectrum (the startup itself is all midrange, which fogs the music), never in the chord's key.
halo        her voice an octave up (WORLD: same formants, doubled pitch), like the Quest 'success' chime leaping up
            an octave. Fed only into the reverb, so it hangs above her as a shimmer, not a second singer.
dream       vocal + reverb(vocal + halo), the reverb kept between the music's body and the hiss (550-3500 Hz) and
            contained to her phrases (it rings while she sings, then eases out), so her ambience does not bleed into
            the music and her breath never becomes steam.

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


def startup_ir(x: np.ndarray, sr: int, length: float = 2.0, color_db: float = 6.0, smooth_oct: float = 0.5,
               air_db_oct: float = 0.0, predelay: float = 0.025, seed: int = 0) -> np.ndarray:
    """Reverb with the startup sound's swell-and-fade shape (its level, frame by frame) on a full-range, airy
    spectrum. `color_db` caps how much of the startup's own (smoothed) spectrum is kept: the startup is near-pure
    tones at 300-1000 Hz, so taking all of it makes a midrange fog that sits right in the music. `air_db_oct` tilts
    everything above 1 kHz; keep it at 0 or below: lifting the top turns her breath into steam-engine hiss."""
    x = np.asarray(x, dtype=np.float64)[:int(length * sr)]
    nfft, hop = 2048, 512
    win = np.hanning(nfft)
    frames = [x[i:i + nfft] * win for i in range(0, max(1, len(x) - nfft), hop)]
    mag = np.abs(np.fft.rfft(np.array(frames), axis=1)).T                  # (bins, frames)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    level = np.sqrt((mag ** 2).mean(0))                                    # the swell and fade
    shape = _smooth_bins(mag.mean(1, keepdims=True), freqs, smooth_oct)[:, 0]
    shape_db = np.clip(20 * np.log10(np.maximum(shape, 1e-12) / max(1e-12, shape.max())), -color_db, 0.0)
    shape = 10 ** (shape_db / 20)
    shape *= 10 ** (air_db_oct * np.log2(np.maximum(freqs, 1000.0) / 1000.0) / 20)
    shape[freqs > 16000] *= 0.0
    rng = np.random.default_rng(seed)
    spec = shape[:, None] * level[None, :] * np.exp(2j * np.pi * rng.random(mag.shape))
    ir = np.zeros(len(frames) * hop + nfft)
    for k in range(spec.shape[1]):
        ir[k * hop:k * hop + nfft] += np.fft.irfft(spec[:, k], nfft) * win
    ir = sosfilt(butter(2, 120.0, "high", fs=sr, output="sos"), ir)
    ir = np.concatenate([np.zeros(int(predelay * sr)), ir])
    return (ir / max(1e-12, np.sqrt((ir ** 2).sum()))).astype(np.float32)


def halo(x: np.ndarray, sr: int, ratio: float = 2.0, breath: float = 0.15) -> np.ndarray:
    """The vocal at `ratio` times the pitch with the same spectral envelope (WORLD), kept tonal (aperiodicity scaled
    by `breath`, so it rings like a chime instead of hissing) and limited to 1.2-4 kHz."""
    import pyworld as pw
    x = np.asarray(x, dtype=np.float64)
    f0, t = pw.harvest(x, sr, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    ap = np.clip(ap * breath, 0.0, 1.0)
    y = pw.synthesize(f0 * ratio, sp, ap, sr, frame_period=5.0)[:len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    return sosfilt(butter(2, [1200.0, 4000.0], "band", fs=sr, output="sos"), y).astype(np.float32)


def contain(wet: np.ndarray, dry: np.ndarray, sr: int, hold: float = 0.5, release: float = 1.2,
            floor_db: float = -40.0) -> np.ndarray:
    """Let the reverb ring while she sings and for `hold` seconds after, then fade it over `release` seconds, so her
    ambience stays on her voice instead of hanging over the music in the gaps."""
    n = len(wet)
    d = np.zeros(n)
    d[:len(dry)] = np.abs(np.asarray(dry, dtype=np.float64))
    w = max(1, int(0.01 * sr))
    k = len(d) // w
    lvl = np.sqrt((np.pad(d, (0, (k + 1) * w - n)) ** 2).reshape(-1, w).mean(1))
    on = lvl > lvl.max() * 10 ** (floor_db / 20)
    h, r = int(hold * sr / w), max(1, int(release * sr / w))
    g = np.zeros(len(on))
    since = 10 ** 9
    for i, o in enumerate(on):
        since = 0 if o else since + 1
        g[i] = 1.0 if since <= h else 0.5 + 0.5 * np.cos(np.pi * min(1.0, (since - h) / r))
    g = np.convolve(g, np.ones(3) / 3, "same")                 # no zipper steps
    return wet * np.repeat(g, w)[:n]


def dream(x: np.ndarray, sr: int, ir: np.ndarray, wet_db: float = -7.0, halo_db: float = -3.0,
          halo_sig: np.ndarray | None = None, wet_hp: float = 550.0, wet_lp: float = 3500.0,
          contained: bool = True) -> np.ndarray:
    """Dry vocal plus the startup reverb of (vocal + octave halo). The reverb is kept above `wet_hp` (out of the
    music's body) and contained to her phrases. Output is longer by the reverb tail."""
    x = np.asarray(x, dtype=np.float64)
    send = x.copy()
    if halo_sig is not None:
        send = send + np.asarray(halo_sig, dtype=np.float64)[:len(x)] * (_rms(x) / _rms(halo_sig)) * 10 ** (halo_db / 20)
    wet = fftconvolve(send, ir)
    if wet_hp:                                                 # out of the music's body
        wet = sosfilt(butter(4, wet_hp, "high", fs=sr, output="sos"), wet)
    if wet_lp:                                                 # and below the hiss: breath reverb = steam
        wet = sosfilt(butter(4, wet_lp, "low", fs=sr, output="sos"), wet)
    if contained:
        wet = contain(wet, x, sr)
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
    ap.add_argument("--wet", type=float, default=-7.0, help="reverb level vs the dry vocal, dB")
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
