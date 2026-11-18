"""Render a UTAU project (.ust) with a voicebank - a small UTAU-style engine for testing banks.

For every note: look the lyric up in the bank's oto.ini, re-pitch the sample with the WORLD vocoder to the note's pitch
curve (Mode2 pitch bends PBS/PBW/PBY and VBR vibrato), keep the consonant (scaled by Velocity) and stretch the rest to the
note length, apply Intensity and Envelope, then overlap the notes using preutterance/overlap (with UTAU's auto-fit when
the previous note is short).

    python -m voice2utau.sing song.ust BANK_DIR -o song.wav
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio

SR = audio.SR_BANK
FP = 5.0                                  # WORLD frame period, ms
TICKS = 480                               # per quarter note
LYRIC_FALLBACK = {"wo": "o", "を": "お", "づ": "ず", "ぢ": "じ", "ゔ": "ぶ", "ゐ": "い", "ゑ": "え"}
SILENT = {"R", "r", "", "っ", "ッ", "・"}


@dataclass
class Note:
    lyric: str
    length: int                           # ticks
    notenum: int
    tempo: float
    pre: float | None = None
    ovl: float | None = None
    intensity: float = 100.0
    velocity: float = 100.0
    envelope: list[str] = field(default_factory=list)
    pbs: str = ""
    pbw: list[float] = field(default_factory=list)
    pby: list[float] = field(default_factory=list)
    vbr: list[float] = field(default_factory=list)
    start: float = 0.0                    # seconds
    dur: float = 0.0                      # seconds


def _f(v: str, default: float | None = None) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _floats(v: str) -> list[float]:
    return [x for x in (_f(s) for s in v.split(",")) if x is not None] if v else []


def parse_ust(path: Path) -> list[Note]:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("cp932")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", "replace")
    tempo, notes, cur = 120.0, [], None
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        m = re.match(r"^\[#(.+)\]$", line)
        if m:
            tag = m.group(1)
            cur = {} if re.fullmatch(r"\d+", tag) else None
            if cur is not None:
                notes.append(cur)
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if cur is None:
            if k == "Tempo":
                tempo = _f(v, tempo)
        else:
            cur[k] = v
    out, t = [], 0.0
    for d in notes:
        tempo = _f(d.get("Tempo"), tempo)
        n = Note(lyric=d.get("Lyric", "R").strip(), length=int(_f(d.get("Length"), 480)), notenum=int(_f(d.get("NoteNum"), 60)),
                 tempo=tempo, pre=_f(d.get("PreUtterance")), ovl=_f(d.get("VoiceOverlap")),
                 intensity=_f(d.get("Intensity"), 100.0), velocity=_f(d.get("Velocity"), 100.0),
                 envelope=d.get("Envelope", "").split(",") if d.get("Envelope") else [],
                 pbs=d.get("PBS", ""), pbw=_floats(d.get("PBW", "")), pby=_floats(d.get("PBY", "")), vbr=_floats(d.get("VBR", "")))
        n.start, n.dur = t, n.length / TICKS * 60.0 / tempo
        t += n.dur
        out.append(n)
    return out


def read_oto(bank: Path) -> dict[str, tuple[str, list[float]]]:
    oto: dict[str, tuple[str, list[float]]] = {}
    for p in [bank / "oto.ini", *sorted(bank.glob("*/oto.ini"))]:
        if not p.exists():
            continue
        raw = p.read_bytes()
        try:
            text = raw.decode("cp932")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", "replace")
        for line in text.replace("\r", "").split("\n"):
            if "=" not in line:
                continue
            wav, rest = line.split("=", 1)
            f = rest.split(",")
            if len(f) < 6:
                continue
            vals = [_f(x, 0.0) for x in f[1:6]]
            oto.setdefault(f[0], (str(p.parent / wav), vals))
    return oto


def resolve(lyric: str, oto: dict) -> str | None:
    if lyric in oto:
        return lyric
    hira = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in lyric)
    for cand in (hira, LYRIC_FALLBACK.get(hira, ""), hira.split(" ")[-1], LYRIC_FALLBACK.get(hira.split(" ")[-1], "")):
        if cand and cand in oto:
            return cand
    return None


def pitch_curve(notes: list[Note], total_s: float) -> np.ndarray:
    """Pitch in cents (MIDI*100) per WORLD frame for the whole song."""
    nfr = int(total_s * 1000 / FP) + 2
    cents = np.zeros(nfr)
    for n in notes:
        a, b = int(n.start * 1000 / FP), int((n.start + n.dur) * 1000 / FP) + 1
        cents[a:b] = n.notenum * 100
    prev = None
    for n in notes:
        base = n.notenum * 100
        if n.lyric not in SILENT and n.pbw:
            parts = n.pbs.split(";")
            x0 = _f(parts[0], 0.0)
            y0 = _f(parts[1]) if len(parts) > 1 and parts[1] != "" else None
            if y0 is None:
                y0 = (prev.notenum * 100 - base) / 10.0 if prev is not None and prev.lyric not in SILENT else 0.0
            ts, ys = [x0], [y0]
            for i, w in enumerate(n.pbw):
                ts.append(ts[-1] + w)
                ys.append(n.pby[i] if i < len(n.pby) else 0.0)
            t_abs = n.start * 1000 + np.array(ts)
            fa, fb = max(0, int(t_abs[0] / FP)), min(nfr, int(t_abs[-1] / FP) + 1)
            for fi in range(fa, fb):
                tm = fi * FP
                k = int(np.clip(np.searchsorted(t_abs, tm) - 1, 0, len(t_abs) - 2))
                u = (tm - t_abs[k]) / max(1e-6, t_abs[k + 1] - t_abs[k])
                u = 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))                    # S-curve, UTAU's default
                cents[fi] = base + 10.0 * (ys[k] + (ys[k + 1] - ys[k]) * u)
        if n.lyric not in SILENT and len(n.vbr) >= 3:
            vlen, cyc, depth = n.vbr[0], max(n.vbr[1], 10.0), n.vbr[2]
            fin = n.vbr[3] if len(n.vbr) > 3 else 0.0
            fout = n.vbr[4] if len(n.vbr) > 4 else 0.0
            ph = n.vbr[5] if len(n.vbr) > 5 else 0.0
            L = n.dur * 1000 * vlen / 100.0
            s0 = (n.start + n.dur) * 1000 - L
            for fi in range(int(s0 / FP), int((n.start + n.dur) * 1000 / FP)):
                u = (fi * FP - s0) / max(1.0, L)
                g = min(1.0, u / max(1e-3, fin / 100.0)) if fin else 1.0
                g = min(g, (1 - u) / max(1e-3, fout / 100.0)) if fout else g
                if 0 <= fi < nfr:
                    cents[fi] += depth * max(0.0, g) * np.sin(2 * np.pi * ((fi * FP - s0) / cyc + ph / 100.0))
        if n.lyric not in SILENT:
            prev = n
    return cents


def envelope(n: Note, L: int) -> np.ndarray:
    """UTAU envelope p1,p2,p3,v1,v2,v3,v4[,%,p4,p5,v5] (ms / %) as a gain curve of L samples."""
    e = n.envelope
    if len(e) < 7:
        env = np.ones(L)
        k = min(L, int(0.01 * SR))
        env[:k] = np.linspace(0, 1, k)
        env[-k:] *= np.linspace(1, 0, k)
        return env
    p1, p2, p3 = (_f(e[i], 0.0) for i in range(3))
    v1, v2, v3, v4 = (_f(e[i], 100.0) for i in range(3, 7))
    p4 = _f(e[8], 0.0) if len(e) > 8 else 0.0
    p5 = _f(e[9], 0.0) if len(e) > 9 else 0.0
    v5 = _f(e[10], 100.0) if len(e) > 10 else 100.0
    Lms = L / SR * 1000
    ts = [0, p1, p1 + p2, p1 + p2 + p5, Lms - p4 - p3, Lms - p4, Lms]
    vs = [0, v1, v2, v5 if p5 else v2, v3, v4, 0]
    order = np.argsort(np.clip(ts, 0, Lms), kind="stable")
    t_arr, v_arr = np.clip(np.array(ts)[order], 0, Lms), np.array(vs)[order]
    return np.interp(np.arange(L) / SR * 1000, t_arr, v_arr) / 100.0


class Engine:
    def __init__(self, bank: Path, keep_level: bool = False, consonant_cap_db: float | None = None,
                 engine: str = "world"):
        """keep_level: every note keeps the vowel loudness of its source sample (WORLD's output level swings with the
        pitch shift, by up to ~13 dB at +2.5 octaves). consonant_cap_db: no 10 ms of a note's consonant may be louder
        than its own rendered vowel plus this many dB (None = off)."""
        import pyworld  # noqa: F401
        self.bank = Path(bank)
        self.oto = read_oto(self.bank)
        self.cache: dict[tuple, tuple] = {}
        self.keep_level, self.cap_db = keep_level, consonant_cap_db
        if engine not in ("world", "psola"):
            raise ValueError("engine must be world or psola")
        self.engine = engine
        self.ref_level: dict[tuple, float] = {}

    def source_vowel_rms(self, wav: str, off: float, cons: float, cut: float) -> float:
        key = (wav, off, cut)
        if key not in self.ref_level:
            x, sr = audio.read_wav(Path(wav))
            a = int((off + cons) / 1000 * sr) + int(0.03 * sr)
            seg = x[a:a + int(0.2 * sr)]
            self.ref_level[key] = float(np.sqrt((seg ** 2).mean())) if len(seg) else 0.0
        return self.ref_level[key]

    def analyse(self, wav: str, off: float, cut: float):
        key = (wav, off, cut)
        if key not in self.cache:
            import pyworld as pw
            x, sr = audio.read_wav(Path(wav))
            if sr != SR:
                x = audio.resample(x, sr, SR)
            a = int(off / 1000 * SR)
            b = a + int(-cut / 1000 * SR) if cut < 0 else (len(x) - int(cut / 1000 * SR) if cut > 0 else len(x))
            seg = x[a:max(b, a + 1024)].astype("float64")
            f0, t = pw.dio(seg, SR, frame_period=FP, f0_floor=70, f0_ceil=800)
            f0 = pw.stonemask(seg, f0, t, SR)
            f0 = fill_from_frq(f0, seg, Path(wav), a)
            sp = pw.cheaptrick(seg, f0, t, SR)
            ap = pw.d4c(seg, f0, t, SR)
            self.cache[key] = (f0, sp, ap, seg)
        return self.cache[key]

    def render(self, notes: list[Note], tail_s: float = 1.0) -> tuple[np.ndarray, dict]:
        import pyworld as pw
        total = (notes[-1].start + notes[-1].dur if notes else 0) + tail_s
        cents = pitch_curve(notes, total)
        out = np.zeros(int(total * SR) + SR)
        missing: dict[str, int] = {}
        voiced = [i for i, n in enumerate(notes) if n.lyric not in SILENT]
        regions: list[tuple[float, float, float]] = []
        for j, i in enumerate(voiced):
            n = notes[i]
            alias = resolve(n.lyric, self.oto)
            if alias is None:
                missing[n.lyric] = missing.get(n.lyric, 0) + 1
                continue
            wav, (off, cons, cut, opre, oovl) = self.oto[alias]
            pre = n.pre if n.pre is not None else opre
            ovl = n.ovl if n.ovl is not None else oovl
            prev = notes[i - 1] if i > 0 else None
            if prev is not None and prev.lyric not in SILENT and pre - ovl > prev.dur * 1000 / 2:  # UTAU auto-fit
                r = (prev.dur * 1000 / 2) / max(1e-6, pre - ovl)
                pre, ovl = pre * r, ovl * r
            nxt = notes[i + 1] if i + 1 < len(notes) else None
            L_ms = n.dur * 1000 + pre
            if nxt is not None and nxt.lyric not in SILENT:
                na = resolve(nxt.lyric, self.oto)
                if na:
                    _, (_, _, _, npre, novl) = self.oto[na]
                    npre = nxt.pre if nxt.pre is not None else npre
                    novl = nxt.ovl if nxt.ovl is not None else novl
                    if npre - novl > n.dur * 1000 / 2:
                        r = (n.dur * 1000 / 2) / max(1e-6, npre - novl)
                        npre, novl = npre * r, novl * r
                    L_ms += novl - npre
            else:
                L_ms += 30.0
            f0, sp, ap, seg = self.analyse(wav, off, cut)
            nin = len(f0)
            vel = 2 ** ((100 - n.velocity) / 100)
            cons_in = min(nin - 2, max(1, int(cons / FP)))
            cons_out = int(cons * vel / FP)
            nout = max(cons_out + 2, int(L_ms / FP))
            idx = np.empty(nout)
            k = min(cons_out, nout)
            idx[:k] = np.linspace(0, cons_in, k, endpoint=False) if k else []
            idx[k:] = np.linspace(cons_in, nin - 1, nout - k) if nout > k else []
            ii = np.clip(np.round(idx).astype(int), 0, nin - 1)
            t0 = n.start - pre / 1000
            fr = np.clip(((t0 * 1000 + np.arange(nout) * FP) / FP).astype(int), 0, len(cents) - 1)
            hz = 440.0 * 2 ** ((cents[fr] / 100 - 69) / 12)
            nf0 = np.where(f0[ii] > 0, hz, 0.0)
            if self.engine == "psola":
                y = psola(seg, f0, idx, nf0, nout)
            else:
                y = pw.synthesize(np.ascontiguousarray(nf0), np.ascontiguousarray(sp[ii]), np.ascontiguousarray(ap[ii]), SR, FP)
            if self.keep_level or self.cap_db is not None:
                from .synthbank import cap_consonant
                c_end = int(cons_out * FP / 1000 * SR)                       # end of the (velocity-scaled) consonant
                v0 = c_end + int(0.03 * SR)
                vow = y[v0:v0 + int(0.2 * SR)]
                if len(vow) > int(0.04 * SR):
                    if self.keep_level:
                        ref, now = self.source_vowel_rms(wav, off, cons, cut), float(np.sqrt((vow ** 2).mean()))
                        if ref > 0 and now > 0:
                            y = y * float(np.clip(ref / now, 10 ** (-12 / 20), 10 ** (18 / 20)))
            y = y * envelope(n, len(y)) * (n.intensity / 100.0)
            if self.cap_db is not None:
                # after the envelope, so the consonant is compared with the vowel as it will be heard
                c_end = int(cons_out * FP / 1000 * SR)
                v0 = c_end + int(0.03 * SR)
                v1 = min(len(y), int((pre + n.dur * 1000) / 1000 * SR) - int(0.02 * SR))   # this note's own vowel
                if c_end > 0 and v1 - v0 > int(0.04 * SR):
                    y = _cap_to(y, c_end, v0, v1, self.cap_db)
            fi = min(len(y), int(max(ovl, 3.0) / 1000 * SR))                       # cross-fade in over the overlap
            y[:fi] *= np.linspace(0, 1, fi)
            s = int(t0 * SR)
            if s < 0:
                y, s = y[-s:], 0
            out[s:s + len(y)] += y[:len(out) - s]
            regions.append((n.start - pre / 1000 + ovl / 1000, n.start, n.start + n.dur))
        if self.cap_db is not None:
            # final pass on the summed track: each note's consonant (after the previous note's overlap) may not be
            # louder than that note's vowel as it actually sounds in context
            for c0, c1, e in regions:
                a, b = int((c0 + 0.002) * SR), int(c1 * SR)
                v0, v1 = int((c1 + 0.03) * SR), int((e - 0.02) * SR)
                if b - a > int(0.01 * SR) and v1 - v0 > int(0.04 * SR) and a >= 0:
                    seg = out[a:v1].copy()
                    out[a:v1] = _cap_to(seg, b - a, v0 - a, v1 - a, self.cap_db)
        peak = float(np.abs(out).max()) or 1.0
        return (out / peak * 0.89).astype("float32"), {"notes": len(voiced), "missing": missing, "seconds": round(total, 1)}


def _cap_to(y: np.ndarray, c_end: int, v0: int, v1: int, cap_db: float) -> np.ndarray:
    """Limit y[:c_end] so no 10 ms is louder than the median 10 ms level of y[v0:v1] + cap_db."""
    w = int(0.01 * SR)
    vow = y[v0:v1]
    nv = len(vow) // w
    if nv < 3:
        return y
    target = float(np.median(np.sqrt((vow[:nv * w].reshape(nv, w) ** 2).mean(1)))) * 10 ** (cap_db / 20)
    hop = w // 2
    starts = np.arange(0, max(1, c_end - w + 1), hop)
    lvl = np.array([np.sqrt((y[s:s + w] ** 2).mean()) for s in starts])
    g = np.minimum(1.0, target / np.maximum(lvl, 1e-12))
    gain = np.interp(np.arange(c_end), starts + w // 2, g)
    k = int(0.004 * SR)
    gain = np.minimum(gain, np.convolve(gain, np.ones(k) / k, "same"))
    y = y.copy()
    y[:c_end] *= gain
    return y


def read_frq(path: Path) -> tuple[int, np.ndarray] | None:
    """UTAU .frq: (hop in samples, f0 per hop) or None."""
    import struct
    try:
        b = Path(path).read_bytes()
        if b[:8] != b"FREQ0003":
            return None
        hop = struct.unpack_from("<i", b, 8)[0]
        n = struct.unpack_from("<i", b, 36)[0]
        d = np.frombuffer(b, dtype="<f8", count=2 * n, offset=40)
        return hop, d[0::2].copy()
    except (OSError, struct.error, ValueError):
        return None


def fill_from_frq(f0: np.ndarray, seg: np.ndarray, wav: Path, offset: int) -> np.ndarray:
    """Where pitch detection found no voicing in a frame that is clearly voiced (loud, energy mostly under 2 kHz), take
    the pitch from the sample's .frq file - what UTAU resamplers use. Breathy or console-tone voices (synthesised at a
    known pitch) otherwise lose whole stretches of pitch tracking and come out monotone."""
    frq = read_frq(wav.with_name(wav.stem + "_wav.frq"))
    if frq is None:
        return f0
    hop, fq = frq
    f0 = f0.copy()
    n = 1024
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    lo = freqs < 2000                                       # voiced sounds: most energy below 2 kHz; hiss is above
    frame = np.arange(len(f0)) * FP / 1000 * SR
    e = np.array([float(np.sqrt(np.mean(seg[max(0, int(c) - n // 2):int(c) + n // 2] ** 2))) if int(c) < len(seg) else 0.0
                  for c in frame])
    loud = e > e.max() * 10 ** (-30 / 20) if len(e) and e.max() > 0 else np.zeros(len(f0), bool)
    for i in np.where((f0 <= 0) & loud)[0]:
        c = int(frame[i])
        x = seg[max(0, c - n // 2):c + n // 2]
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        X = np.abs(np.fft.rfft(x * win)) ** 2
        if X[lo].sum() > 0.5 * X.sum():
            j = min(len(fq) - 1, max(0, (offset + c) // max(1, hop)))
            if fq[j] > 0:
                f0[i] = fq[j]
    return f0


def psola(x: np.ndarray, f0: np.ndarray, idx: np.ndarray, target_hz: np.ndarray, nout: int) -> np.ndarray:
    """TD-PSOLA: re-pitch/stretch by overlap-adding two-period Hann grains cut at the sample's own pitch marks.

    x: source segment; f0: its WORLD f0 per FP frame (0 = unvoiced); idx: input frame position for each output frame;
    target_hz: output pitch per output frame (0 = unvoiced). Keeps the waveform detail that a vocoder smooths away."""
    fps = FP / 1000 * SR
    nf = len(f0)
    marks, pos = [], 0.0
    while pos < len(x):                                         # pitch marks: one per period, 5 ms apart when unvoiced
        f = f0[min(int(pos / fps), nf - 1)]
        marks.append(pos)
        pos += SR / f if f > 0 else 0.005 * SR
    marks = np.array(marks)
    L = int(nout * fps)
    out = np.zeros(L + 4096)
    wsum = np.zeros(L + 4096)
    frames = np.arange(nout)
    t = 0.0
    while t < L:
        j = t / fps
        ti = float(np.interp(j, frames, idx)) * fps
        k = int(np.clip(np.searchsorted(marks, ti), 0, len(marks) - 1))
        if k > 0 and abs(marks[k - 1] - ti) < abs(marks[k] - ti):
            k -= 1
        m = marks[k]
        fin = f0[min(int(m / fps), nf - 1)]
        ft = float(np.interp(j, frames, target_hz))
        Tin = SR / fin if fin > 0 else 0.005 * SR
        Tout = SR / ft if (fin > 0 and ft > 0) else Tin
        h = int(Tin)
        a, b = int(m) - h, int(m) + h
        if a >= 0 and b <= len(x) and h > 1:
            win = np.hanning(2 * h)
            c = int(t) - h
            if c >= 0:
                out[c:c + 2 * h] += x[a:b] * win
                wsum[c:c + 2 * h] += win
        t += max(Tout, 8.0)
    out, wsum = out[:L], wsum[:L]
    y = np.where(wsum > 1e-3, out / np.maximum(wsum, 0.5), out)
    return y


def level_match(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Scale a rendered track so the RMS of its sounding 50 ms windows is target_db (peak scaling let one hot note
    make a whole track ~9 dB quieter than another)."""
    w = int(0.05 * SR)
    n = len(y) // w
    if n == 0:
        return y
    r = np.sqrt((y[:n * w].reshape(n, w) ** 2).mean(1))
    on = r > r.max() * 10 ** (-40 / 20)
    if not on.any():
        return y
    cur = float(np.sqrt((r[on] ** 2).mean()))
    return y * (10 ** (target_db / 20) / max(cur, 1e-9))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ust", type=Path, nargs="+")
    ap.add_argument("bank", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--gain", type=float, nargs="*", help="mix gain per UST in dB (default 0 for the first, -6 for others)")
    ap.add_argument("--keep-level", action="store_true", help="keep each note's vowel at its sample's loudness at any pitch")
    ap.add_argument("--engine", choices=["world", "psola"], default="world",
                    help="world: vocoder resynthesis (smooth); psola: moves the sample's own cycles (keeps its exact timbre)")
    ap.add_argument("--consonant-cap", type=float, default=None, metavar="DB",
                    help="limit each note's consonant to its vowel level + DB (e.g. -1)")
    ap.add_argument("--transpose", type=int, default=0, help="shift every note by this many semitones (e.g. -12)")
    ap.add_argument("--suffix", default="", help="sing every lyric with this alias suffix when the bank has it "
                    "(e.g. a voice colour like _sam)")
    a = ap.parse_args()
    eng = Engine(a.bank, keep_level=a.keep_level, consonant_cap_db=a.consonant_cap, engine=a.engine)
    mix = None
    for k, u in enumerate(a.ust):
        notes = parse_ust(u)
        for n in notes:
            n.notenum += a.transpose
            if a.suffix and n.lyric not in SILENT:
                base = resolve(n.lyric, eng.oto)
                if base and base + a.suffix in eng.oto:
                    n.lyric = base + a.suffix
        y, info = eng.render(notes)
        y = level_match(y)                     # each track scaled by its singing loudness, not by its single loudest peak
        g = (a.gain[k] if a.gain and k < len(a.gain) else (0.0 if k == 0 else -6.0))
        y = y * 10 ** (g / 20)
        mix = y if mix is None else np.pad(mix, (0, max(0, len(y) - len(mix)))) + np.pad(y, (0, max(0, len(mix) - len(y))))
        print(f"{u.name}: {info['notes']} notes, {info['seconds']} s, missing lyrics {info['missing']}")
    mix = mix / max(1e-9, float(np.abs(mix).max())) * 0.89
    audio.write_wav(a.out, mix.astype("float32"), SR)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
