"""Binaural headphone placement: put a mono voice in front of the listener and let her drift slowly through the frontal
field, the way a Quest UI panel floats in front of you.

Per ear: a head-shadow filter (Brown & Duda's one-pole/one-zero spherical-head model) and the matching
arrival delay (the interaural time difference), plus two early reflections from the sides so the voice sits outside the
head instead of between the ears. Applied on overlapping blocks, so the position can move.

    python -m voice2utau.spatial IN.wav OUT.wav [--width 35] [--period 24] [--center 0]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HEAD_RADIUS = 0.0875          # metres
C = 343.0                     # speed of sound, m/s
# (delay s, gain dB, azimuth offset deg): a balanced pair either side, slightly different delays so the ears decorrelate
REFLECTIONS = ((0.0060, -12.0, 70.0), (0.0071, -12.0, -70.0), (0.0113, -17.0, 115.0), (0.0128, -17.0, -115.0))


def ear_response(freqs: np.ndarray, az: float) -> tuple[np.ndarray, np.ndarray]:
    """Complex (left, right) responses for a source at azimuth `az` radians (0 = straight ahead, + = right)."""
    w = 2 * np.pi * freqs
    w0 = C / HEAD_RADIUS
    out = []
    for ear in (-np.pi / 2, np.pi / 2):
        theta = abs(np.angle(np.exp(1j * (az - ear))))       # angle between the source and this ear's axis, 0..pi
        alpha = 1.05 + 0.95 * np.cos(theta * 180.0 / 150.0)
        shadow = (1 + 1j * alpha * w / (2 * w0)) / (1 + 1j * w / (2 * w0))
        if theta < np.pi / 2:
            tau = -HEAD_RADIUS / C * np.cos(theta)
        else:
            tau = HEAD_RADIUS / C * (theta - np.pi / 2)
        tau += HEAD_RADIUS / C                                # keep delays non-negative
        out.append(shadow * np.exp(-1j * w * tau))
    return out[0], out[1]


def drift(n: int, sr: int, width_deg: float = 35.0, period_s: float = 24.0, center_deg: float = 0.0) -> np.ndarray:
    """Azimuth (radians) per sample: a slow sway across the front, starting centred."""
    t = np.arange(n) / sr
    return np.radians(center_deg + width_deg * np.sin(2 * np.pi * t / period_s))


def binaural(x: np.ndarray, sr: int, az: np.ndarray | float = 0.0, block: int = 4096,
             reflections=REFLECTIONS) -> np.ndarray:
    """Mono -> (n, 2) stereo for headphones. `az`: one azimuth (radians) or one per sample."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    az = np.broadcast_to(np.asarray(az, dtype=np.float64), (n,))
    hop = block // 2
    win = np.sqrt(np.hanning(block + 1)[:block])              # sqrt-Hann in and out: overlap-adds to 1 at hop N/2
    pad = np.concatenate([np.zeros(hop), x, np.zeros(block)])
    out = np.zeros((len(pad), 2))
    freqs = np.fft.rfftfreq(block, 1 / sr)
    w = 2 * np.pi * freqs
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(pad) - block + 1, hop):
        mid = min(n - 1, start)                               # block centre, in input samples
        key = int(round(np.degrees(az[mid]) * 4))             # quarter-degree steps
        if key not in cache:
            a = np.radians(key / 4)
            hl, hr = ear_response(freqs, a)
            for d, g_db, off in reflections:
                rl, rr = ear_response(freqs, a + np.radians(off))
                g = 10 ** (g_db / 20) * np.exp(-1j * w * d)
                hl, hr = hl + g * rl, hr + g * rr
            cache[key] = hl, hr
        hl, hr = cache[key]
        spec = np.fft.rfft(pad[start:start + block] * win)
        out[start:start + block, 0] += np.fft.irfft(spec * hl, block) * win
        out[start:start + block, 1] += np.fft.irfft(spec * hr, block) * win
    y = out[hop:hop + n]
    rms_in = np.sqrt((x ** 2).mean()) + 1e-12
    rms_out = np.sqrt((y ** 2).mean()) + 1e-12
    return (y * rms_in / rms_out).astype(np.float32)


def main() -> None:
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--width", type=float, default=35.0, help="sway either side of centre, degrees")
    ap.add_argument("--period", type=float, default=24.0, help="seconds for one full sway")
    ap.add_argument("--center", type=float, default=0.0, help="centre azimuth, degrees (+ = right)")
    a = ap.parse_args()
    x, sr = sf.read(a.inp, dtype="float32", always_2d=True)
    x = x.mean(1)
    y = binaural(x, sr, drift(len(x), sr, a.width, a.period, a.center))
    sf.write(a.out, y, sr, subtype="PCM_16" if a.out.suffix.lower() == ".wav" else None)
    print(f"wrote {a.out} ({len(y) / sr:.1f} s, sway ±{a.width:g}° every {a.period:g} s)")


if __name__ == "__main__":
    main()
