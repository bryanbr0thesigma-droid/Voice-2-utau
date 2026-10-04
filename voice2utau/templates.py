"""Template voicebanks used to fill the morae the uploaded voice does not cover.

Two sources:
  * EspeakTemplate     - generated locally with espeak-ng. No licence concerns, robotic timbre
                         (which is fine: the RVC model replaces the timbre).
  * VoicebankTemplate  - any existing UTAU voicebank (folder or .zip with oto.ini), e.g. a
                         free-to-use bank you have chosen. YOU are responsible for checking
                         that its licence allows this use.
"""
from __future__ import annotations

import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

from . import audio
from .extract import Clip
from .ingest import safe_extract_zip
from .morae import MORAE, VOICELESS_CLASSES, DEFAULT_PRE_MS, Mora

# measured espeak-ng "ja" median F0 for -p (pitch) 0/25/50/75/99
_P_POINTS = [0, 25, 50, 75, 99]
_F0_BY_VOICE = {"ja": [62, 75, 95, 125, 164], "ja+f3": [100, 125, 195, 255, 330]}


def pick_espeak_voice(target_hz: float | None) -> tuple[str, int]:
    """Choose an espeak voice/pitch whose natural F0 is as close to target_hz as possible."""
    if not target_hz:
        return "ja", 50
    best = None
    for voice, f0s in _F0_BY_VOICE.items():
        if voice != "ja" and target_hz < 140:      # female formants make no sense for a deep target
            continue
        for p in range(0, 100, 5):
            f0 = float(np.interp(p, _P_POINTS, f0s))
            err = abs(np.log2(f0 / target_hz))
            if best is None or err < best[0]:
                best = (err, voice, p)
    return best[1], best[2]


class TemplateError(RuntimeError):
    pass


class EspeakTemplate:
    name = "espeak-ng (generated)"
    license = "Generated locally with espeak-ng (GPL-3 tool; output audio is yours)"

    def __init__(self, target_f0: float | None = None, speed: int = 110):
        self.voice, self.pitch = pick_espeak_voice(target_f0)
        self.speed = speed
        try:
            subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise TemplateError("espeak-ng is not installed (apt install espeak-ng)") from e

    def _say(self, text: str) -> tuple[np.ndarray, int]:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t.wav"
            r = subprocess.run(["espeak-ng", "-v", self.voice, "-s", str(self.speed), "-p",
                                str(self.pitch), "-w", str(out), text], capture_output=True)
            if r.returncode != 0 or not out.exists():
                raise TemplateError(f"espeak-ng failed for {text!r}: {r.stderr.decode(errors='ignore')[:200]}")
            x, sr = audio.read_wav(out)
        return audio.resample(x, sr, audio.SR_BANK), audio.SR_BANK

    def get(self, mora: Mora) -> Clip | None:
        sr = audio.SR_BANK
        if mora.key == "nn":
            x, _ = self._say("あん")
            a, b = audio.trim_silence(x, sr)
            x = x[a:b]
            x = _nasal_tail(x, sr)
            if x is None:
                return None
            return _finish(mora, x, 0.02, "template", self.name)
        x, _ = self._say(mora.kana)
        a, b = audio.trim_silence(x, sr)
        x = x[a:b]
        if len(x) < sr * 0.1:
            return None
        return _finish(mora, x, None, "template", self.name)


def _nasal_tail(x: np.ndarray, sr: int) -> np.ndarray | None:
    """Cut the nasal murmur off the end of a synthesised 'あん'."""
    from scipy.signal import butter, sosfilt
    hp = sosfilt(butter(4, 1500, "high", fs=sr, output="sos"), x)
    e, eh = audio.rms_envelope(x, sr, 10), audio.rms_envelope(hp, sr, 10)
    ratio = eh / (e + 1e-9)
    bright = np.where(ratio > 0.2)[0]
    if len(bright) == 0:
        return None
    start = max(0, (int(bright[-1]) - 3) * int(sr * 0.01))
    tail = x[start:]
    return tail if len(tail) >= sr * 0.12 else None


def _finish(mora: Mora, x: np.ndarray, split_s: float | None, source: str, origin: str) -> Clip:
    sr = audio.SR_BANK
    x = audio.fade(x, sr)
    if split_s is None:
        split_s = estimate_split(mora, x, sr)
    f0 = audio.median_f0(x[len(x) // 3:], sr)
    return Clip(mora, x, split_s, f0, source, 1.0, origin)


def estimate_split(mora: Mora, x: np.ndarray, sr: int) -> float:
    """Consonant/vowel boundary (seconds) for a clip we have no alignment for."""
    dur = len(x) / sr
    default = DEFAULT_PRE_MS.get(mora.cls, 60) / 1000
    if mora.cls in VOICELESS_CLASSES:
        onset = audio.voicing_onset(x, sr)
        if onset is not None and 0.03 <= onset <= dur * 0.7:
            return onset
    return min(default, dur * 0.5)


# ------------------------------------------------------------ real UTAU banks

_KANA_BY_ALIAS = {m.kana: m for m in MORAE.values()}
_KEY_ALIAS = {m.key: m for m in MORAE.values()}
_KEY_ALIAS.update({"si": MORAE["shi"], "ti": MORAE["chi"], "tu": MORAE["tsu"], "hu": MORAE["fu"],
                   "zi": MORAE["ji"], "n": MORAE["nn"], "N": MORAE["nn"]})


def _to_hiragana(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in unicodedata.normalize("NFKC", s))


def alias_to_mora(alias: str) -> Mora | None:
    a = _to_hiragana(alias.strip())
    a = a[2:] if a.startswith("- ") else a
    if a in _KANA_BY_ALIAS:
        return _KANA_BY_ALIAS[a]
    return _KEY_ALIAS.get(a.lower())


def _read_text(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


class VoicebankTemplate:
    """A CV-style UTAU voicebank on disk (folder, or a .zip that is extracted to `workdir`)."""

    def __init__(self, path: Path, workdir: Path, name: str | None = None):
        path = Path(path)
        if path.suffix.lower() == ".zip":
            dest = Path(workdir) / "template_bank"
            safe_extract_zip(path, dest, allowed_ext={".wav", ".ini", ".txt", ".frq", ".png", ".bmp"})
            path = dest
        otos = sorted(path.rglob("oto.ini"), key=lambda p: len(p.parts))
        if not otos:
            raise TemplateError(f"no oto.ini found in template {path}")
        self.root = otos[0].parent
        self.name = name or self.root.name
        self.license = "user-supplied (verify the template's terms of use)"
        self._entries: dict[str, tuple[Path, float, float, float, float]] = {}
        for line in _read_text(otos[0]).splitlines():
            if "=" not in line:
                continue
            wav, rest = line.split("=", 1)
            f = rest.split(",")
            if len(f) < 6:
                continue
            m = alias_to_mora(f[0])
            if m is None or m.key in self._entries:
                continue
            try:
                offset, _cons, cutoff, pre = float(f[1]), float(f[2]), float(f[3]), float(f[4])
            except ValueError:
                continue
            wp = self.root / wav.strip().replace("\\", "/")
            if wp.exists():
                self._entries[m.key] = (wp, offset, cutoff, pre, 0.0)

    def available(self) -> set[str]:
        return set(self._entries)

    def get(self, mora: Mora) -> Clip | None:
        e = self._entries.get(mora.key)
        if not e:
            return None
        wp, offset, cutoff, pre, _ = e
        x, sr = audio.read_wav(wp)
        x = audio.resample(x, sr, audio.SR_BANK)
        sr = audio.SR_BANK
        a = int(offset / 1000 * sr)
        if cutoff < 0:
            b = a + int(-cutoff / 1000 * sr)
        elif cutoff > 0:
            b = len(x) - int(cutoff / 1000 * sr)
        else:
            b = len(x)
        x = x[a:max(a + 1, min(b, len(x)))]
        if len(x) < sr * 0.08:
            return None
        return _finish(mora, x, max(0.0, pre / 1000), "template", self.name)
