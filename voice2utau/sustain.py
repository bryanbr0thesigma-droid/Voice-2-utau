"""Longer, steadier vowels for a finished English bank, using held notes from singing recordings.

1. `find_held` looks through a vocal recording for held vowels: a vowel the recogniser labels, whose voiced,
   spectrally steady run lasts >= MIN_HELD_S.
2. `Body` = the steadiest part of such a run, pitch-flattened to the bank's pitch.
3. `extend` inserts a body into the vowel of a unit (CV clip or lone vowel) at a steady point, crossfaded and
   phase-aligned, so UTAU has a long steady vowel to stretch instead of a few hundred ms of speech.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import audio
from .crossling import read_oto
from .english import classify

HOP_S = 0.01
MIN_HELD_S = 0.30
MIN_HNR_DB = 11.2          # 10th percentile of solo speech in the villager corpus
MIN_CONF = 0.5
STEADY_DIST = 4.5          # max distance (z-scored MFCC, 50 ms smoothed) from the vowel's centre for a frame to count as "same vowel"
MAX_BODY_S = 0.8
MAX_ROUGH = 10.0           # about the median roughness of the bank's own recorded vowels (5-18)
XFADE_MS = 30.0
DIPHTHONG_NUCLEUS = {"ay": "aa", "aw": "aa", "ey": "eh", "ow": "ao", "oy": "ao"}   # vowel held while a glide waits


@dataclass
class Body:
    vowel: str
    audio: np.ndarray             # 44.1 kHz, flattened, steady
    source: str                   # 'song' or 'rvc'
    conf: float = 1.0
    hnr: float = 0.0
    f0: float = 0.0
    origin: str = ""
    mfcc: np.ndarray | None = field(default=None, repr=False)


def _nearest(t_src: np.ndarray, v: np.ndarray, t: np.ndarray, fill: float) -> np.ndarray:
    idx = np.clip(np.searchsorted(t_src, t), 0, len(t_src) - 1)
    out = v[idx].astype(np.float64)
    out[np.abs(t_src[idx] - t) > 0.02] = fill
    return out


def mfcc_frames(x: np.ndarray, sr: int) -> np.ndarray:
    """(frames, 13) MFCCs at a 10 ms hop; frame i is centred at i * 10 ms."""
    import torch
    import torchaudio
    m = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13, melkwargs=dict(
        n_fft=2048, hop_length=int(sr * HOP_S), n_mels=40, f_max=8000.0))
    return m(torch.tensor(x, dtype=torch.float32)).T.numpy()


def voicing_frames(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per 10 ms frame: harmonics-to-noise ratio (dB), f0 (Hz, 0 = unvoiced), level (dB re peak)."""
    import parselmouth
    n = len(x) // int(sr * HOP_S) + 1
    g = np.arange(n) * HOP_S
    h = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr).to_harmonicity(
        time_step=HOP_S, minimum_pitch=75.0)
    hnr = _nearest(h.xs(), h.values[0], g, -200.0)
    tp, f0 = audio.f0_track(x, sr)
    f0g = _nearest(tp, f0, g, 0.0)
    env = audio.rms_envelope(x, sr, 10.0)
    lvl = 20 * np.log10(np.maximum(_nearest(np.arange(len(env)) * HOP_S, env, g, 0.0), 1e-6) / max(float(env.max()), 1e-6))
    return hnr, f0g, lvl


def find_held(x: np.ndarray, sr: int, phones, min_conf: float = MIN_CONF) -> list[dict]:
    """Held vowels in a recording. `phones` are the recogniser's Phone list for the same file.
    Returns dicts: vowel, start, end (s), conf, hnr, f0, f0_std (semitones), mfcc (mean of the run)."""
    hnr, f0, lvl = voicing_frames(x, sr)
    mf = mfcc_frames(x, sr)
    n = min(len(hnr), len(mf))
    z = (mf[:n, 1:13] - mf[:n, 1:13].mean(0)) / (mf[:n, 1:13].std(0) + 1e-6)
    z = uniform_filter1d(z, 5, axis=0)       # raw 10 ms MFCCs are too jittery at high pitch
    voiced = (hnr[:n] >= MIN_HNR_DB) & (f0[:n] > 0) & (lvl[:n] > -35.0)
    out, taken = [], np.zeros(n, bool)
    for p in sorted(phones, key=lambda p: -p.conf):
        s = classify(p.sym)
        if s.kind != "V" or p.conf < min_conf:
            continue
        c = min(n - 1, int((p.start + p.end) / 2 / HOP_S))
        if not voiced[c] or taken[c]:
            continue
        anchor = z[max(0, c - 2):c + 3].mean(0)
        a = b = c
        while a - 1 >= 0 and voiced[a - 1] and np.linalg.norm(z[a - 1] - anchor) < STEADY_DIST:
            a -= 1
        while b + 1 < n and voiced[b + 1] and np.linalg.norm(z[b + 1] - anchor) < STEADY_DIST:
            b += 1
        if (b - a + 1) * HOP_S < MIN_HELD_S:
            continue
        taken[a:b + 1] = True
        v = f0[a:b + 1]
        v = v[v > 0]
        out.append({"vowel": s.value, "start": a * HOP_S, "end": (b + 1) * HOP_S, "conf": p.conf,
                    "hnr": float(np.median(hnr[a:b + 1])), "f0": float(np.median(v)),
                    "f0_std": float(np.std(12 * np.log2(v / np.median(v)))), "sym": p.sym,
                    "mfcc": z[a:b + 1, :].mean(0)})
    return sorted(out, key=lambda d: d["start"])


# ---------------------------------------------------------------------------- bodies

def _flat(x: np.ndarray, sr: int, hz: float, max_st: float = 9.0) -> np.ndarray:
    y, _ = audio.flatten_pitch(x, sr, hz, max_semitones=max_st)
    return y


ENV_HZ = np.geomspace(200, 5000, 20)


def env_frames(x: np.ndarray, sr: int, order: int = 14) -> np.ndarray:
    """(frames, 20) LPC spectral envelope in dB at log-spaced frequencies, level removed; frame i is centred at
    i * 10 ms. Smoother than MFCCs at high pitch, where harmonics are wider apart than the formants."""
    from scipy.linalg import solve_toeplitz
    y = audio.resample(x, sr, 16000) if sr != 16000 else x
    y = np.pad(y, (200, 200))
    w = np.hanning(400)
    freqs = np.fft.rfftfreq(1024, 1 / 16000)
    out = []
    for i in range(0, len(y) - 400 - 160, 160):
        f = y[i:i + 400] * w
        if np.abs(f).max() < 1e-4:
            out.append(np.zeros(len(ENV_HZ)))
            continue
        r = np.correlate(f, f, "full")[399:399 + order + 1]
        r[0] *= 1.0001
        a = np.concatenate([[1.0], -solve_toeplitz(r[:order], r[1:order + 1])])
        e = np.interp(ENV_HZ, freqs, -20 * np.log10(np.maximum(np.abs(np.fft.rfft(a, 1024)), 1e-9)))
        out.append(e - e.mean())
    return np.array(out)


def drift(x: np.ndarray, sr: int) -> float:
    """Envelope distance between the first and last third of a clip: how much the vowel changes while held."""
    m = env_frames(x, sr)
    k = max(1, len(m) // 3)
    return float(np.linalg.norm(m[:k].mean(0) - m[-k:].mean(0)))


def song_bodies(x: np.ndarray, sr: int, held: list[dict], bank_f0: float) -> list[Body]:
    """Steady bodies (trimmed 40 ms at both ends, at most MAX_BODY_S, pitch-flattened) from `find_held` results."""
    out = []
    for h in held:
        a, b = h["start"] + 0.04, h["end"] - 0.04
        if b - a > MAX_BODY_S:
            mid = (a + b) / 2
            a, b = mid - MAX_BODY_S / 2, mid + MAX_BODY_S / 2
        seg = _flat(x[int(a * sr):int(b * sr)], sr, bank_f0)
        body = Body(h["vowel"], seg, "song", h["conf"], h["hnr"], bank_f0, f"{a:.2f}-{b:.2f}s")
        if roughness(seg, sr) <= MAX_ROUGH:
            out.append(body)
    return out


def espeak_vowel(vowel: str, hz: float, seconds: float = 0.9) -> np.ndarray | None:
    """A long, flat espeak-ng vowel (steady middle of the vowel, lengthened with PSOLA) — RVC input."""
    import subprocess
    import tempfile
    from pathlib import Path

    import parselmouth
    from parselmouth.praat import call

    from .english import _ESPEAK_V
    from .templates import pick_espeak_voice
    sr = audio.SR_BANK
    voice, pitch = pick_espeak_voice(hz, "en")
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "v.wav"
        subprocess.run(["espeak-ng", "-v", voice, "-s", "80", "-p", str(pitch), "-w", str(w),
                        f"[[{_ESPEAK_V[vowel]}]]"], check=True, capture_output=True)
        x, s = audio.read_wav(w)
    x = audio.resample(x, s, sr)
    a, b = audio.trim_silence(x, sr)
    x = x[a:b]
    if len(x) < sr * 0.12:
        return None
    k = int(len(x) * 0.25)
    core = x[k:len(x) - k]
    snd = parselmouth.Sound(core.astype(np.float64), sampling_frequency=sr)
    factor = max(1.0, seconds / snd.duration)
    y = call(snd, "Lengthen (overlap-add)", 75, 600, factor)
    return np.asarray(y.values[0], dtype=np.float32)


def rvc_bodies(vowels: list[str], bank_f0: float, backend, variants=(1.0, 0.92, 1.08),
               keep: int = 2) -> tuple[list[Body], list[str]]:
    """Sustained vowels made by RVC from espeak-ng vowels (source='rvc'). Each vowel is made at a few input pitches
    because RVC output is sometimes unsteady; the `keep` steadiest windows per vowel that pass MAX_ROUGH are returned."""
    from . import rvc
    from .extract import Clip
    from .morae import Mora
    sr = audio.SR_BANK
    clips = []
    for v in vowels:
        for f in variants:
            y = espeak_vowel(v, bank_f0 * f)
            if y is not None:
                clips.append(Clip(Mora(v, v, "", v, "V", "en"), y, 0.0, bank_f0 * f, "template", 1.0, f"espeak x{f}"))
    done, warns = rvc.convert_clips(clips, backend, 0)
    found: dict[str, list[tuple[float, Body]]] = {}
    for c in done:
        if c.source != "rvc":
            continue
        a, b = audio.trim_silence(c.audio, sr)
        y = c.audio[a:b]
        y = y[int(0.04 * sr):len(y) - int(0.04 * sr)]
        body = steadiest(Body(c.mora.vowel, y, "rvc", 1.0, 0.0, bank_f0, f"{c.origin}->RVC"), sr, 0.4)
        body.audio = _flat(body.audio, sr, bank_f0)
        r = roughness(body.audio, sr)
        if r <= MAX_ROUGH:
            found.setdefault(body.vowel, []).append((r, body))
    return [b for v in found.values() for _, b in sorted(v, key=lambda t: t[0])[:keep]], warns


# ---------------------------------------------------------------------------- joining

def _period(x: np.ndarray, sr: int, hz: float) -> int:
    return max(2, int(round(sr / hz)))


def _best_lag(ref: np.ndarray, cand: np.ndarray, max_lag: int) -> int:
    """Shift (0..max_lag samples dropped from the start of `cand`) that best lines `cand` up with `ref`."""
    n = min(len(ref), len(cand) - max_lag)
    if n <= 8:
        return 0
    best, lag = -2.0, 0
    r = ref[:n] - ref[:n].mean()
    for L in range(max_lag + 1):
        c = cand[L:L + n]
        c = c - c.mean()
        d = float(np.linalg.norm(r) * np.linalg.norm(c)) + 1e-9
        v = float(r @ c) / d
        if v > best:
            best, lag = v, L
    return lag


def _join(a: np.ndarray, b: np.ndarray, n: int, period: int) -> np.ndarray:
    """a followed by b, crossfaded over n samples, b shifted by <= 1 period so the waveforms line up."""
    if len(a) < n or len(b) < n + period:
        return np.concatenate([a, b])
    lag = _best_lag(a[-n:], b, period)
    b = b[lag:]
    fade = np.linspace(0, 1, n, dtype=np.float32)
    return np.concatenate([a[:-n], a[-n:] * (1 - fade) + b[:n] * fade, b[n:]])


@dataclass
class Result:
    audio: np.ndarray
    inserted_ms: float
    join_ms: float
    body: Body
    rel_head: float
    rel_tail: float


def _centre(m: np.ndarray, i: int, w: int = 2) -> np.ndarray:
    return m[max(0, i - w):i + w + 1].mean(0)


def extend(x: np.ndarray, sr: int, pre_ms: float, vowel: str, bodies: dict[str, list[Body]],
           scale: float, bank_f0: float) -> Result | None:
    """Insert a held body into the vowel of clip `x` (vowel starts at `pre_ms`). `scale` = typical envelope distance between different vowels.
    Returns None if the clip has no usable room or no body fits."""
    nucleus = DIPHTHONG_NUCLEUS.get(vowel, vowel)
    glide = nucleus != vowel
    cands = (bodies.get(vowel, []) if glide else []) + bodies.get(nucleus, [])
    if not cands:
        return None
    m = env_frames(x, sr)
    dur_ms = len(x) / sr * 1000
    lo = int((pre_ms + 25) / 10)                       # the vowel starts at the preutterance
    hi = int(((pre_ms + 0.4 * (dur_ms - pre_ms)) if glide else dur_ms - 40) / 10)
    hi = min(hi, len(m) - 3)
    if hi - lo < 1:
        return None
    _, f0, _ = voicing_frames(x, sr)
    best = None
    for body in cands:
        bm = env_frames(body.audio, sr)
        k = max(2, int(0.08 / HOP_S))
        b_head, b_tail = bm[1:k + 1].mean(0), bm[-k - 1:-1].mean(0)
        for j in range(lo, hi + 1):
            if j >= len(f0) or f0[j] <= 0:
                continue
            c = _centre(m, j)
            dh, dt = float(np.linalg.norm(c - b_head)), float(np.linalg.norm(c - b_tail))
            move = float(np.linalg.norm(m[min(len(m) - 1, j + 2)] - m[max(0, j - 2)]))
            score = max(dh, dt) + 0.5 * move
            if best is None or score < best[0]:
                best = (score, j, body, dh / scale, dt / scale)
    if best is None:
        return None
    _, j, body, rh, rt = best
    pos = int(j * HOP_S * sr)
    n = int(sr * XFADE_MS / 1000)
    per = _period(x, sr, bank_f0)
    ref = x[max(0, pos - n):pos + n]
    g_clip = float(np.sqrt(np.mean(ref ** 2)) + 1e-9)
    g_body = float(np.sqrt(np.mean(body.audio[:len(ref)] ** 2)) + 1e-9)
    bd = body.audio * float(np.clip(g_clip / g_body, 0.5, 2.0))
    y = _join(x[:pos], bd, n, per)
    y = _join(y, x[pos:], n, per)
    return Result(y.astype(np.float32), (len(y) - len(x)) / sr * 1000, j * 10.0, body, rh, rt)


def vowel_scale(bank, report_samples) -> tuple[float, dict[str, np.ndarray]]:
    """Typical envelope distance between different vowels in this bank, and each vowel's centroid.
    Built from the vowel part (after the preutterance) of every recorded CV clip."""
    oto = read_oto(bank / "oto.ini")
    acc: dict[str, list[np.ndarray]] = {}
    for key, (alias, p) in oto.items():
        if not key.startswith("cv_") or key not in report_samples:
            continue
        x, sr = audio.read_wav(bank / f"{key}.wav")
        m = env_frames(x, sr)
        a, b = int((p[3] + 10) / 10), int((len(x) / sr * 1000 - 20) / 10)
        if b - a >= 2:
            acc.setdefault(key.split("_")[2], []).append(m[a:b + 1].mean(0))
    cen = {v: np.mean(c, 0) for v, c in acc.items() if len(c) >= 3}
    d = [float(np.linalg.norm(cen[a] - cen[b])) for a in cen for b in cen if a < b]
    return (float(np.median(d)) if d else 20.0), cen          # 20 = typical value when the bank is too small to tell


# ---------------------------------------------------------------------------- bank

def vowel_of(key: str) -> str | None:
    p = key.split("_")
    v = p[2] if key.startswith("cv_") and len(p) == 3 else p[1] if key.startswith("v_") and len(p) == 2 else None
    return v if v in DIPHTHONG_NUCLEUS or v in {"aa", "ae", "ah", "ao", "eh", "er", "ih", "iy", "uh", "uw"} else None


def apply(bank, out, bodies: list[Body], max_rel: float = 1.0, name: str = "VLGR") -> dict:
    """Copy `bank` to `out`, then lengthen the vowel of every CV and lone-vowel unit that a body fits.
    All aliases (English, Japanese, Mandarin) that point at a changed wav get the new length."""
    import shutil
    from pathlib import Path

    from .crossling import _fmt
    bank, out = Path(bank), Path(out)
    rep = json.loads((bank / "voice2utau_report.json").read_text(encoding="utf-8"))
    bank_f0 = float(rep["bank_f0_hz"])
    recorded = {s["key"] for s in rep["samples"] if s["source"] == "recorded"}
    scale, _ = vowel_scale(bank, recorded)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(bank, out)
    by_vowel: dict[str, list[Body]] = {}
    for b in bodies:
        by_vowel.setdefault(b.vowel, []).append(b)
    lines = (out / "oto.ini").read_bytes().decode("cp932").splitlines()
    first: dict[str, list[float]] = {}
    users: dict[str, list[int]] = {}
    for i, l in enumerate(lines):
        if "=" in l:
            wav, rest = l.split("=", 1)
            users.setdefault(wav[:-4], []).append(i)
            first.setdefault(wav[:-4], [float(v) for v in rest.split(",")[1:6]])
    sr = audio.SR_BANK
    units, skipped = [], {"no_vowel": 0, "no_room_or_body": 0, "poor_join": 0}
    for key in sorted(users):
        v = vowel_of(key)
        if v is None:
            continue
        x, _ = audio.read_wav(bank / f"{key}.wav")
        p = first[key]
        r = extend(x, sr, p[3], v, by_vowel, scale, bank_f0)
        if r is None:
            skipped["no_room_or_body"] += 1
            continue
        if max(r.rel_head, r.rel_tail) > max_rel:
            skipped["poor_join"] += 1
            continue
        audio.write_wav(out / f"{key}.wav", r.audio, sr)
        for i in users[key]:
            wav, rest = lines[i].split("=", 1)
            f = rest.split(",")
            q = [float(t) for t in f[1:6]]
            q[1] = min(q[1], r.join_ms)
            q[2] = q[2] - r.inserted_ms if q[2] < 0 else q[2] + r.inserted_ms
            lines[i] = f"{wav}={f[0]},{','.join(_fmt(t) for t in q)}"
        units.append({"key": key, "vowel": v, "body_vowel": r.body.vowel, "body_source": r.body.source,
                      "body_origin": r.body.origin, "inserted_ms": round(r.inserted_ms), "join_ms": round(r.join_ms),
                      "mismatch_head": round(r.rel_head, 2), "mismatch_tail": round(r.rel_tail, 2),
                      "aliases": len(users[key])})
    (out / "oto.ini").write_bytes(("\n".join(lines) + "\n").encode("cp932"))
    summary = {"extended": len(units), "skipped": skipped, "max_mismatch_allowed": max_rel, "scale": round(scale, 2),
               "bodies": [{"vowel": b.vowel, "source": b.source, "origin": b.origin,
                           "seconds": round(len(b.audio) / sr, 2), "drift": round(drift(b.audio, sr) / scale, 2)}
                          for b in bodies], "units": units}
    rep["sustain"] = summary
    (out / "voice2utau_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    ro = out / "readme.txt"
    ro.write_text(ro.read_text(encoding="utf-8") +
                  f"Sustain pass: {len(units)} CV / lone-vowel clips have a held vowel spliced into their vowel "
                  "(sources: held notes from singing recordings, or espeak vowels converted with RVC; see 'sustain' in "
                  "voice2utau_report.json). Japanese/Mandarin entries that reuse those clips got the same longer vowel.\n",
                  encoding="utf-8")
    return summary


def roughness(x: np.ndarray, sr: int) -> float:
    """How un-steady a held vowel is: mean frame-to-frame spectral-envelope change (dB) plus any level dip > 3 dB
    (RVC output sometimes drops out for a few frames)."""
    m = env_frames(x, sr)[2:-2]
    if len(m) < 4:
        return 99.0
    rms = 20 * np.log10(np.maximum(audio.rms_envelope(x, sr, 10.0)[2:-2], 1e-6))
    dip = max(0.0, float(np.median(rms) - rms.min()) - 3.0)
    return float(np.linalg.norm(np.diff(m, axis=0), axis=1).mean() + dip)


def steadiest(b: Body, sr: int, seconds: float) -> Body:
    """The steadiest `seconds`-long window of a body (windows stepped by 50 ms)."""
    n = int(seconds * sr)
    if len(b.audio) <= n:
        return b
    best = min(range(0, len(b.audio) - n + 1, int(0.05 * sr)),
               key=lambda a: roughness(b.audio[a:a + n], sr))
    return Body(b.vowel, b.audio[best:best + n], b.source, b.conf, b.hnr, b.f0,
                f"{b.origin} @{best / sr:.2f}s")


MONOPHTHONGS = ["aa", "ae", "ah", "ao", "eh", "er", "ih", "iy", "uh", "uw"]


def main() -> None:
    import argparse
    import tempfile
    from pathlib import Path

    from . import rvc
    from .phonemes import PhonemeRecognizer
    ap = argparse.ArgumentParser(description="Lengthen the vowels of an English bank with held notes from songs / RVC.")
    ap.add_argument("bank", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True, help="new bank folder (the input bank is not changed)")
    ap.add_argument("--songs", type=Path, nargs="*", default=[], help="vocal recordings (solo singing, background removed)")
    ap.add_argument("--rvc-model", type=Path)
    ap.add_argument("--rvc-index", type=Path)
    ap.add_argument("--rvc-command")
    ap.add_argument("--rvc-all", action="store_true", help="make every vowel with RVC, ignoring the songs")
    ap.add_argument("--max-mismatch", type=float, default=99.0, help="skip units whose join mismatch exceeds this")
    a = ap.parse_args()
    rep = json.loads((a.bank / "voice2utau_report.json").read_text(encoding="utf-8"))
    bank_f0 = float(rep["bank_f0_hz"])
    bodies: list[Body] = []
    if a.songs and not a.rvc_all:
        rec = PhonemeRecognizer()
        with tempfile.TemporaryDirectory() as td:
            for i, src in enumerate(a.songs):
                w44, w16 = Path(td) / f"s{i}_44.wav", Path(td) / f"s{i}_16.wav"
                audio.ffmpeg_decode_to_wav(src, w44, audio.SR_BANK)
                audio.ffmpeg_decode_to_wav(src, w16, audio.SR_REC)
                x, sr = audio.read_wav(w44)
                bodies += song_bodies(x, sr, find_held(x, sr, rec.recognise_file(w16)), bank_f0)
    covered = {b.vowel for b in bodies}
    need = [v for v in MONOPHTHONGS if v not in covered]
    print(f"held vowels from songs: {sorted(covered) or 'none'}; RVC needed for: {need}")
    if need:
        if not a.rvc_model:
            print("no RVC model given: those vowels are left unchanged")
        else:
            rb, warns = rvc_bodies(need, bank_f0, rvc.make_backend(a.rvc_model, a.rvc_index, a.rvc_command))
            bodies += rb
            for w in warns:
                print("warning:", w)
    s = apply(a.bank, a.out, bodies, a.max_mismatch)
    print(f"extended {s['extended']} clips; skipped {s['skipped']}")


if __name__ == "__main__":
    main()
