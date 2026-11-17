"""Kaso's Meta Quest moments, applied to a rendered vocal.

passthrough  her voice as the real world seen through the headset cameras: band envelopes refreshed at a camera
             frame rate (stair-stepped, like low-fps passthrough video), part of each band's detail swapped for sensor
             grain, the colour narrowed (top end rolled off), and a faint late copy (camera lag).
emerge       every phrase starts in passthrough and resolves into her clean voice, like putting the headset on.
guardian     notes at the top of her range "touch the boundary": the Guardian boundary shimmer flares under them.
close_up     a little proximity warmth: she sits right at your ear, like a hand menu.

    python -m voice2utau.quest VOCAL.wav SONG.ust OUT.wav --shimmer boundary_draw_loop.wav
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfilt


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt((np.asarray(x, dtype=np.float64) ** 2).mean()) + 1e-12)


def passthrough(x: np.ndarray, sr: int, fps: float = 36.0, n_bands: int = 14, grain: float = 0.35,
                lag_ms: float = 28.0, lag_db: float = -9.0, top_hz: float = 6500.0, seed: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.astype(np.float32)
    rng = np.random.default_rng(seed)
    hop = max(1, int(sr / fps))
    frame_of = (np.arange(n) // hop) * hop
    smooth = butter(2, 40.0, "low", fs=sr, output="sos")
    xfade = max(1, int(0.002 * sr))
    edges = np.geomspace(90.0, min(9000.0, sr / 2 * 0.9), n_bands + 1)
    out = np.zeros(n)
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = butter(2, [lo, hi], "band", fs=sr, output="sos")
        xb = sosfilt(band, x)
        env = np.sqrt(np.maximum(sosfilt(smooth, xb ** 2), 0.0))
        held = env[frame_of]                                   # refreshed once per camera frame
        held = np.convolve(held, np.ones(xfade) / xfade, "same")
        floor = env.max() * 1e-3 + 1e-12
        gain = np.clip(held / np.maximum(env, floor), 0.0, 4.0)
        nb = sosfilt(band, rng.standard_normal(n))
        nb /= _rms(nb)
        out += xb * gain * (1 - grain) + nb * held * grain
    out = sosfilt(butter(2, top_hz, "low", fs=sr, output="sos"), out)
    d = int(lag_ms / 1000 * sr)
    if 0 < d < n:
        out[d:] += out[:-d] * 10 ** (lag_db / 20)
    return (out * _rms(x) / _rms(out)).astype(np.float32)


def phrases(notes, gap: float = 0.15) -> list[tuple[float, float]]:
    """(start, end) of each run of sung notes; a rest of `gap` seconds or more starts a new phrase."""
    from .sing import SILENT
    out: list[list[float]] = []
    for nt in notes:
        if nt.lyric in SILENT:
            continue
        if out and nt.start - out[-1][1] < gap:
            out[-1][1] = nt.start + nt.dur
        else:
            out.append([nt.start, nt.start + nt.dur])
    return [(a, b) for a, b in out]


def emerge(clean: np.ndarray, pt: np.ndarray, sr: int, notes, ramp: float = 0.4, lead: float = 0.12) -> np.ndarray:
    """Start every phrase in passthrough and cross into the clean voice over `ramp` seconds."""
    w = np.zeros(len(clean))
    for a, _ in phrases(notes):
        i0 = max(0, int((a - lead) * sr))
        i1 = min(len(w), int((a + ramp) * sr))
        if i1 <= i0:
            continue
        t = np.arange(i0, i1) / sr - a                         # seconds from the phrase start
        wv = np.where(t < 0, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip(t / ramp, 0, 1)))
        w[i0:i1] = np.maximum(w[i0:i1], wv)
    return (clean * (1 - w) + pt[:len(clean)] * w).astype(np.float32)


def high_notes(notes, top_percent: float = 5.0, min_dur: float = 0.15):
    """The sung notes in the top `top_percent` of the song's pitches (by note count, so one stray high note can't set
    the bar)."""
    from .sing import SILENT
    sung = [nt for nt in notes if nt.lyric not in SILENT]
    if not sung:
        return []
    nums = np.array([nt.notenum for nt in sung])
    budget = len(nums) * top_percent / 100
    pitches, counts = np.unique(nums, return_counts=True)
    thr, total = pitches[-1], 0
    for p, k in zip(pitches[::-1], counts[::-1]):          # highest pitches first, while they fit the budget
        if total + k > budget and total > 0:
            break
        thr, total = p, total + k
    return [nt for nt in sung if nt.notenum >= thr and nt.dur >= min_dur]


def guardian(vocal: np.ndarray, sr: int, notes, shimmer: np.ndarray, db: float = -12.0,
             attack: float = 0.03, release: float = 0.25) -> np.ndarray:
    """Lay the Guardian boundary shimmer under the notes at the top of her range, following her level."""
    y = np.asarray(vocal, dtype=np.float64).copy()
    sh = sosfilt(butter(2, [1500.0, 9000.0], "band", fs=sr, output="sos"), np.asarray(shimmer, dtype=np.float64))
    if not len(sh) or _rms(sh) < 1e-9:
        return y.astype(np.float32)
    sh /= _rms(sh)
    env = np.sqrt(np.maximum(sosfilt(butter(2, 15.0, "low", fs=sr, output="sos"), y ** 2), 0.0))
    pos = 0
    for nt in high_notes(notes):
        a = max(0, int(nt.start * sr))
        b = min(len(y), int((nt.start + nt.dur + release) * sr))
        if b - a < 2:
            continue
        m = b - a
        seg = np.take(sh, np.arange(pos, pos + m), mode="wrap")
        pos += m
        shape = np.ones(m)
        na, nr = min(m, int(attack * sr)), min(m, int(release * sr))
        shape[:na] = np.linspace(0, 1, na)
        shape[m - nr:] *= np.linspace(1, 0, nr)
        y[a:b] += seg * env[a:b] * shape * 10 ** (db / 20) * np.sqrt(2)
    return y.astype(np.float32)


def close_up(x: np.ndarray, sr: int, gain_db: float = 2.0, f0: float = 160.0) -> np.ndarray:
    """Low shelf (RBJ): the proximity warmth of a voice right next to your ear."""
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    al = np.sin(w) / 2 * np.sqrt(2)
    c = np.cos(w)
    b = [A * ((A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al), 2 * A * ((A - 1) - (A + 1) * c),
         A * ((A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al)]
    a = [(A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al, -2 * ((A - 1) + (A + 1) * c),
         (A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x).astype(np.float32)


def main() -> None:
    import argparse
    import soundfile as sf
    from . import audio
    from .sing import parse_ust
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vocal", type=Path)
    ap.add_argument("ust", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--shimmer", type=Path, help="Guardian boundary sound for the flares")
    ap.add_argument("--no-emerge", action="store_true")
    a = ap.parse_args()
    x, sr = sf.read(a.vocal, dtype="float32")
    notes = parse_ust(a.ust)
    y = close_up(x, sr)
    if not a.no_emerge:
        y = emerge(y, passthrough(y, sr), sr, notes)
    if a.shimmer:
        s, ssr = audio.read_wav(a.shimmer)
        y = guardian(y, sr, notes, audio.resample(s, ssr, sr) if ssr != sr else s)
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * 0.99 / peak
    sf.write(a.out, y, sr, subtype="PCM_16")
    print(f"wrote {a.out}: {len(phrases(notes))} phrases emerge from passthrough, "
          f"{len(high_notes(notes)) if a.shimmer else 0} notes touch the Guardian")


if __name__ == "__main__":
    main()
