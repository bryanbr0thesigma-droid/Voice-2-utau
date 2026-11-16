"""Build a UTAU "sound-effect" voicebank from folders of wav files (e.g. every Windows version's system sounds).

Each distinct sound becomes one alias (`xp_ding`, `win7_afternoon_logon` ...). Sounds that are byte-for-byte the same
audio in several folders are stored once and get one alias per folder. Every clip is one-shot: the oto marks the whole
clip as the unstretched "consonant", so the editor plays it at its natural length instead of stretching it to the note.

Pitch: sound effects have no usable pitch for UTAU's resampler, so every sample gets a .frq file that says "this sample is
a constant C4" - it plays as recorded on C4 and is shifted like a keyed sampler on other notes.

    python -m voice2utau.sfxbank --group win31 DIR --group xp DIR ... -o OUT --name "Windows Sounds"
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio

C4_HZ = 261.6256
SR = audio.SR_BANK
MIN_PEAK = 0.01                 # quieter than this counts as silent


def _safe(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def clean_name(stem: str, group: str) -> str:
    """'Windows XP Ding' (group xp) -> 'ding'; 'Windows Logon' -> 'logon'."""
    n = _safe(stem)
    tag = _safe(group)
    for _ in range(2):
        for pre in ("windows_", f"{tag}_", f"{tag.removeprefix('win')}_", "win_"):
            if len(pre) > 1 and n.startswith(pre) and len(n) > len(pre):
                n = n[len(pre):]
    return n or "sound"


def theme_of(path: Path, group: str) -> str:
    m = re.match(r"^([A-Za-z0-9 ]+?)_win\d+_", path.parent.name)
    return _safe(m.group(1)) if m else ""


def read_mono(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    return x.mean(1), sr


def write_frq(path: Path, n_samples: int, hz: float = C4_HZ, hop: int = 256) -> None:
    """UTAU .frq: 'FREQ0003', hop, mean f0, 4 zero ints, count, then (f0, amplitude) double pairs."""
    n = max(1, n_samples // hop)
    with open(path, "wb") as f:
        f.write(b"FREQ0003" + struct.pack("<i", hop) + struct.pack("<d", hz) + b"\0" * 16 + struct.pack("<i", n))
        f.write(struct.pack("<2d", hz, 0.5) * n)


def onset_ms(x: np.ndarray, sr: int, frac: float = 0.1, cap_ms: float = 80.0) -> float:
    idx = np.nonzero(np.abs(x) >= frac * np.abs(x).max())[0]
    return float(min((idx[0] / sr * 1000) if len(idx) else 0.0, cap_ms))


def build(groups: list[tuple[str, Path]], out_dir: Path, name: str, peak_db: float = -3.0) -> dict:
    """groups: [(tag, folder)] in priority order (the first folder to contain a sound owns its file)."""
    root = out_dir / name
    root.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}                         # content hash -> entry
    skipped = {"unreadable": [], "silent": []}
    aliases: dict[str, str] = {}                       # alias -> file stem
    order: list[tuple[str, str, str]] = []
    for tag, folder in groups:
        for p in sorted(Path(folder).rglob("*")):
            if p.suffix.lower() != ".wav":
                continue
            got = read_mono(p)
            if got is None:
                skipped["unreadable"].append(str(p))
                continue
            x, sr = got
            if len(x) < 64 or float(np.abs(x).max()) < MIN_PEAK:
                skipped["silent"].append(str(p))
                continue
            h = hashlib.sha1(np.round(x * 32767).astype("int16").tobytes()).hexdigest()
            theme = theme_of(p, tag)
            base = "_".join(s for s in (_safe(tag), theme, clean_name(p.stem, tag)) if s)
            if h in seen:
                e = seen[h]
                if base not in aliases and base not in e["aliases"]:
                    e["aliases"].append(base)
                    e["groups"].append(tag)
                continue
            seen[h] = {"aliases": [base], "groups": [tag], "src": str(p), "x": x, "sr": sr}
            order.append((h, tag, base))
    entries, lines, used = [], [], set()
    for h, tag, base in order:
        e = seen[h]
        stem = base
        i = 2
        while stem in used:
            stem, i = f"{base}_{i}", i + 1
        used.add(stem)
        x, sr = e["x"], e["sr"]
        if sr != SR:
            x = audio.resample(x, sr, SR)
        a, b = audio.trim_silence(x, SR, rel_db=-50.0, pad_ms=10.0)          # drop leading/trailing silence
        x = x[a:b] if b - a > 64 else x
        x = x / max(1e-9, float(np.abs(x).max())) * 10 ** (peak_db / 20)
        audio.write_wav(root / f"{stem}.wav", x.astype("float32"), SR)
        write_frq(root / f"{stem}_wav.frq", len(x))
        dur = len(x) / SR * 1000
        pre = onset_ms(x, SR)
        als = []
        for a in e["aliases"]:
            k, j = a, 2
            while k in aliases:
                k, j = f"{a}_{j}", j + 1
            aliases[k] = stem
            als.append(k)
            lines.append(f"{stem}.wav={k},0,{max(dur - 10, 1):.1f},{-dur:.1f},{pre:.1f},0")
        entries.append({"file": f"{stem}.wav", "aliases": als, "groups": e["groups"], "seconds": round(dur / 1000, 2),
                        "source": Path(e["src"]).name})
    (root / "oto.ini").write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932", "replace"))
    (root / "character.txt").write_bytes(f"name={name}\r\nimage=\r\nauthor=\r\nweb=\r\n".encode("cp932", "replace"))
    report = {"name": name, "sounds": len(entries), "aliases": len(lines), "skipped": skipped,
              "groups": [t for t, _ in groups], "entries": entries}
    (root / "sfxbank_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (root / "readme.txt").write_text(
        f"{name}\nSound-effect voicebank built with voice2utau.sfxbank: {len(entries)} distinct sounds, {len(lines)} aliases.\n"
        "Each alias is a whole one-shot clip (type the alias as the lyric). Samples are keyed at C4.\n"
        "Aliases are '<group>_<name>' (win7 also has the theme, e.g. win7_afternoon_logon). Sounds shared by several groups "
        "are one file with several aliases.\nSee sfxbank_report.json for the full list.\n"
        "The sounds belong to their original owners (for Windows: Microsoft; the Windows 98 set on archive.org is a fan recreation). "
        "Personal use only - check the owners' terms before sharing this bank.\n", encoding="utf-8")
    return report


def zip_dir(root: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root.parent))
    return zip_path


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group", nargs=2, action="append", metavar=("TAG", "DIR"), required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--name", default="Sound Effects")
    a = ap.parse_args()
    r = build([(t, Path(d)) for t, d in a.group], a.out, a.name)
    print(f"{r['sounds']} sounds, {r['aliases']} aliases; skipped {len(r['skipped']['unreadable'])} unreadable, "
          f"{len(r['skipped']['silent'])} silent")


if __name__ == "__main__":
    main()
