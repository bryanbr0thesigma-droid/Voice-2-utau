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
_F0_BY_VOICE = {
    "ja": {"ja": [62, 75, 95, 125, 164], "ja+f3": [100, 125, 195, 255, 330]},
    # en-us: p0/p50/p99 measured at 63/88/157 Hz and +f3 at ~176 Hz (p50); other points interpolated
    "en": {"en-us": [63, 74, 88, 120, 157], "en-us+f3": [125, 148, 176, 235, 314]},
}


def pick_espeak_voice(target_hz: float | None, lang: str = "ja") -> tuple[str, int]:
    """Choose an espeak voice/pitch whose natural F0 is as close to target_hz as possible."""
    voices = _F0_BY_VOICE[lang]
    base = next(iter(voices))
    if not target_hz:
        return base, 50
    best = None
    for voice, f0s in voices.items():
        if voice != base and target_hz < 140:      # female formants make no sense for a deep target
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

    def __init__(self, target_f0: float | None = None, speed: int = 110, lang: str = "ja"):
        self.lang = lang
        self.voice, self.pitch = pick_espeak_voice(target_f0, lang)
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
        if mora.lang == "en":
            from .english import espeak_phonemes
            x, _ = self._say(f"[[{espeak_phonemes(mora)}]]")
            a, b = audio.trim_silence(x, sr)
            x = x[a:b]
            if len(x) < sr * 0.1:
                return None
            x, split = _crop_en(mora, x, sr)
            return _finish(mora, x, split, "template", self.name)
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


def _crop_en(mora: Mora, x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Shorten a synthesised syllable to the length of a normal recorded unit; return (audio, split)."""
    split = estimate_split(mora, x, sr)
    glide = 0.15 if mora.vowel in ("aw", "ay", "ey", "ow", "oy") else 0.0   # diphthongs need time to glide
    if mora.kind == "CV":
        x = x[: int((split + 0.20 + glide) * sr)]
    elif mora.kind == "VC":
        a = max(0, int((split - 0.18 - glide) * sr))
        x, split = x[a:], split - a / sr
    else:
        x = x[: int(0.35 * sr)]
    return x, split


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
    part = x[: 2 * len(x) // 3] if mora.kind == "VC" else x[len(x) // 3:]
    f0 = audio.median_f0(part, sr)
    return Clip(mora, x, split_s, f0, source, 1.0, origin)


def estimate_split(mora: Mora, x: np.ndarray, sr: int) -> float:
    """Consonant/vowel boundary (seconds) for a clip we have no alignment for."""
    dur = len(x) / sr
    if mora.lang == "en":
        from .english import VOICELESS, default_split
        if mora.kind == "CV" and mora.cls in VOICELESS:
            onset = audio.voicing_onset(x, sr)
            if onset is not None and 0.03 <= onset <= dur * 0.7:
                return onset
        return default_split(mora, dur)
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


def alias_to_mora(alias: str, units: dict[str, Mora] | None = None) -> Mora | None:
    """Map an oto.ini alias to a unit of the target inventory (Japanese by default)."""
    if units is not None and next(iter(units.values())).lang == "en":
        a = " ".join(alias.lower().split())
        by_alias = {m.kana: m for m in units.values()}
        return by_alias.get(a) or by_alias.get(a[2:] if a.startswith("- ") and f"- {a[2:]}" not in by_alias else a)
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

    def __init__(self, path: Path, workdir: Path, name: str | None = None,
                 units: dict[str, Mora] | None = None):
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
            m = alias_to_mora(f[0], units)
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
