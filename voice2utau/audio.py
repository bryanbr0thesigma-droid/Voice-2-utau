"""Audio helpers: decoding, chunking, pitch analysis, pitch flattening, trimming."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR_BANK = 44100   # UTAU voicebank sample rate
SR_REC = 16000    # phoneme recogniser sample rate


class AudioError(RuntimeError):
    pass


def ffmpeg_decode_to_wav(src: Path, dst: Path, sr: int) -> None:
    """Decode any audio/video file to mono 16-bit PCM wav at `sr` using ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1",
           "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise AudioError("ffmpeg is not installed or not on PATH") from e
    if r.returncode != 0 or not dst.exists():
        raise AudioError(f"ffmpeg could not decode {src.name}: {r.stderr.strip()[:300]}")


def read_wav(path: Path, start: float = 0.0, dur: float | None = None) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        f.seek(max(0, int(start * sr)))
        frames = -1 if dur is None else int(dur * sr)
        x = f.read(frames, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def rms_envelope(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * hop_ms / 1000))
    n = len(x) // hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    frames = x[: n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


def chunk_boundaries(x16: np.ndarray, sr: int, target_s: float = 20.0, search_s: float = 3.0) -> list[tuple[int, int]]:
    """Split a long signal into chunks of ~target_s, cutting at the quietest nearby point."""
    n = len(x16)
    target = int(target_s * sr)
    if n <= int(target * 1.4):
        return [(0, n)]
    env = rms_envelope(x16, sr, 10.0)
    hop = int(sr * 0.01)
    bounds, start = [], 0
    while n - start > int(target * 1.4):
        center = start + target
        lo = max(start + target // 2, center - int(search_s * sr))
        hi = min(n, center + int(search_s * sr))
        seg = env[lo // hop: hi // hop]
        cut = lo + int(np.argmin(seg)) * hop if len(seg) else center
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


def split_utterances(x: np.ndarray, sr: int, max_len_s: float = 15.0, min_len_s: float = 1.0,
                     silence_db: float = -40.0, min_gap_s: float = 0.35) -> list[tuple[float, float]]:
    """Silence-based utterance segmentation (used to export an RVC training dataset)."""
    env = rms_envelope(x, sr, 10.0)
    peak = float(np.percentile(env, 99.5)) + 1e-9
    voiced = env > peak * 10 ** (silence_db / 20)
    segs, start, last_voiced = [], None, None
    gap_frames = int(min_gap_s / 0.01)
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            last_voiced = i
        elif start is not None and i - last_voiced > gap_frames:
            segs.append((start * 0.01, (last_voiced + 1) * 0.01))
            start = None
    if start is not None:
        segs.append((start * 0.01, (last_voiced + 1) * 0.01))
    # merge short pieces, split overlong ones
    out: list[tuple[float, float]] = []
    for a, b in segs:
        if out and (b - out[-1][0]) <= max_len_s and (a - out[-1][1]) < 1.0 and (out[-1][1] - out[-1][0]) < min_len_s:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    final: list[tuple[float, float]] = []
    for a, b in out:
        while b - a > max_len_s:
            final.append((a, a + max_len_s))
            a += max_len_s
        if b - a >= min_len_s:
            final.append((a, b))
    return [(max(0.0, a - 0.1), b + 0.1) for a, b in final]


# ---------------------------------------------------------------------- pitch

def f0_track(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 700.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, f0) with 0 for unvoiced frames."""
    import parselmouth
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    # octave_cost 0.1 (Praat default 0.01): on short sub-regions the default sometimes locks onto
    # the sub-harmonic (e.g. 115 Hz for a 230 Hz voice); measured 0 errors vs 3-10 % before.
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax,
                            voicing_threshold=0.5, octave_cost=0.1)
    f0 = pitch.selected_array["frequency"].copy()
    t = pitch.xs()
    return t, f0


def median_f0(x: np.ndarray, sr: int) -> float | None:
    if len(x) < sr * 0.05:
        return None
    try:
        _, f0 = f0_track(x, sr)
    except Exception:
        return None
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) >= 5 else None


def voicing_onset(x: np.ndarray, sr: int, min_run_ms: float = 40.0) -> float | None:
    """Time (s) of the first stable voiced run, i.e. where a voiceless consonant gives way to the vowel."""
    try:
        t, f0 = f0_track(x, sr)
    except Exception:
        return None
    need = int(min_run_ms / 10)
    run = 0
    for i, f in enumerate(f0):
        run = run + 1 if f > 0 else 0
        if run >= need:
            return float(t[i - need + 1])
    return None


def flatten_pitch(x: np.ndarray, sr: int, target_hz: float, max_semitones: float = 7.0) -> tuple[np.ndarray, bool]:
    """Monotonise voiced regions to `target_hz` with PSOLA (Praat). Returns (audio, changed)."""
    import parselmouth
    from parselmouth.praat import call
    try:
        snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
        med = median_f0(x, sr)
        if med is None:
            return x, False
        if abs(12 * np.log2(target_hz / med)) > max_semitones:
            return x, False
        manip = call(snd, "To Manipulation", 0.01, 60, 700)
        flat = call("Create PitchTier", "flat", 0.0, snd.xmax)
        call(flat, "Add point", 0.0, target_hz)
        call(flat, "Add point", snd.xmax, target_hz)
        call([manip, flat], "Replace pitch tier")
        out = call(manip, "Get resynthesis (overlap-add)")
        y = np.asarray(out.values[0], dtype=np.float32)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        return y[: len(x)], True
    except Exception:
        return x, False


# ------------------------------------------------------------------- trimming

def trim_silence(x: np.ndarray, sr: int, rel_db: float = -32.0, pad_ms: float = 8.0) -> tuple[int, int]:
    """Return (start, end) sample indices of the non-silent core."""
    env = rms_envelope(x, sr, 5.0)
    hop = int(sr * 0.005)
    thr = env.max() * 10 ** (rel_db / 20)
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return 0, len(x)
    pad = int(sr * pad_ms / 1000)
    return max(0, idx[0] * hop - pad), min(len(x), (idx[-1] + 1) * hop + pad)


def fade(x: np.ndarray, sr: int, in_ms: float = 4.0, out_ms: float = 12.0) -> np.ndarray:
    y = x.copy()
    a, b = int(sr * in_ms / 1000), int(sr * out_ms / 1000)
    if a > 0 and len(y) > a:
        y[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b > 0 and len(y) > b:
        y[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return y


def normalise_level(x: np.ndarray, target_rms_db: float = -20.0, max_gain_db: float = 18.0,
                    peak_limit: float = 0.89) -> np.ndarray:
    """Scale so the loud (vowel) part sits near target RMS, never clipping."""
    n = max(1, int(len(x) * 0.5))
    loud = np.sort(np.abs(x))[-n:]
    rms = float(np.sqrt(np.mean(loud ** 2))) + 1e-9
    gain_db = np.clip(target_rms_db - 20 * np.log10(rms), -max_gain_db, max_gain_db)
    y = x * 10 ** (gain_db / 20)
    peak = float(np.abs(y).max())
    if peak > peak_limit:
        y *= peak_limit / peak
    return y.astype(np.float32)
