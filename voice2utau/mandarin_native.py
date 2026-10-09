"""Native Mandarin syllables in the bank's own voice - no English or Japanese units involved.

`mandarin.py` builds pinyin out of English clips (so it carries an English accent). This module instead takes
genuine Mandarin from espeak-ng (cmn-latn-pinyin: aspirated vs unaspirated stops, retroflex zh/ch/sh/r, ü, apical
vowels, nasal codas) and converts it to the character's timbre with an RVC model - all ~410 toneless pinyin
syllables, not only the ones English cannot supply.

Two steps (a trained RVC model of the character is needed for the second):

    python -m voice2utau.mandarin_native dataset BANK [BANK ...] -o train.zip    # training audio for RVC
    python -m voice2utau.mandarin_native build --bank BANK --rvc-model char.pth -o out --into BANK

Limits: the voice model has only ever heard the character's own language(s); sounds it never produced (retroflexes,
ü) come out in its timbre with espeak's articulation, not with a native speaker's. Samples are tone 1 (flat); tones 2-4
come from the pitch curve in the editor. espeak renders the alveolo-palatals j/q/x as post-alveolars (close, not exact).
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from . import audio, rvc
from .crossling import oto_line
from .extract import Clip
from .mandarin import syllables
from .rvcfill import zh_template

SUFFIX = "_zh"          # appended to a pinyin alias that is already taken by another sound in the target bank


def read_aliases(bank: Path) -> set[str]:
    oto = bank / "oto.ini"
    if not oto.exists():
        return set()
    text = oto.read_bytes()
    try:
        text = text.decode("cp932")
    except UnicodeDecodeError:
        text = text.decode("utf-8", "replace")
    return {l.split("=", 1)[1].split(",")[0] for l in text.splitlines() if "=" in l}


def estimate_f0(bank: Path, n: int = 40) -> float:
    """Median F0 over an evenly spaced sample of the wavs under `bank`."""
    files = sorted(p for p in bank.rglob("*.wav") if p.name.lower() != "sample.wav")
    if not files:
        raise ValueError(f"no wav files under {bank}")
    step = max(1, len(files) // n)
    f0s = []
    for p in files[::step][:n]:
        x, sr = audio.read_wav(p)
        f = audio.median_f0(x, sr)
        if f:
            f0s.append(f)
    if not f0s:
        raise ValueError("could not measure any pitch in the bank")
    return float(np.median(f0s))


def templates(f0: float, only: list[str] | None = None) -> list[Clip]:
    clips = []
    for name in sorted(syllables()):
        if only and name not in only:
            continue
        c = zh_template(name, f0)
        if c is not None:
            clips.append(c)
    return clips


def build(out_dir: Path, name: str, f0: float, backend: rvc.RVCBackend | None, *, allow_template: bool = False,
          taken: set[str] | None = None, only: list[str] | None = None) -> Path:
    """Write a stand-alone Mandarin bank `<out_dir>/<name>/`. Without a backend nothing is produced unless
    `allow_template` (then the raw espeak voice is used, for previewing only)."""
    if backend is None and not allow_template:
        raise rvc.RVCError("no RVC backend: Mandarin would not be in the character's voice "
                           "(pass an RVC model, or --preview-template for the raw espeak voice)")
    sr = audio.SR_BANK
    clips = templates(f0, only)
    warns: list[str] = []
    transpose = 0
    if backend is not None and clips:
        tf0 = [c.f0 for c in clips if c.f0]
        transpose = int(np.clip(round(12 * np.log2(f0 / np.median(tf0))), -12, 12)) if tf0 else 0
        clips, warns = rvc.convert_clips(clips, backend, transpose)
    taken = taken or set()
    root = out_dir / name
    root.mkdir(parents=True, exist_ok=True)
    lines, samples = [], []
    for c in clips:
        y, flat = audio.flatten_pitch(c.audio, sr, f0)
        y = audio.normalise_level(y)
        dur = len(y) / sr * 1000
        pre = min(max(c.split_s * 1000, 15.0), dur * 0.6)
        params = [0.0, min(pre + 45, dur - 10), -dur, pre, max(5.0, pre / 3)]
        alias = c.mora.key + (SUFFIX if c.mora.key in taken else "")
        wav = f"zh_{c.mora.key}"
        audio.write_wav(root / f"{wav}.wav", y, sr)
        lines.append(oto_line(wav, alias, params))
        samples.append({"alias": alias, "file": f"{wav}.wav", "source": c.source, "flattened": bool(flat),
                        "duration_ms": round(dur), "preutterance_ms": round(pre)})
    (root / "oto.ini").write_bytes(("\n".join(lines) + "\n").encode("cp932"))
    (root / "character.txt").write_bytes(f"name={name}\nimage=\nauthor=\nweb=\n".encode("cp932"))
    (root / "readme.txt").write_text(
        f"{name}\nNative Mandarin (plain toneless pinyin aliases, tone-1 flat samples; tones come from the pitch curve).\n"
        f"Syllables come from espeak-ng cmn-latn-pinyin"
        + (", converted to the character's voice with RVC.\n" if backend else " (raw, NOT the character's voice - preview only).\n")
        + (f"Aliases already used by another sound in the target bank carry the suffix '{SUFFIX}'.\n" if taken else ""),
        encoding="utf-8")
    renamed = [s["alias"] for s in samples if s["alias"].endswith(SUFFIX)]
    report = {"kind": "mandarin_native", "bank_f0_hz": round(f0, 1), "backend": backend.name if backend else None,
              "transpose": transpose, "syllables": len(samples),
              "converted": sum(s["source"] == "rvc" for s in samples),
              "unconverted_template": [s["alias"] for s in samples if s["source"] != "rvc"] if backend else "all",
              "renamed_aliases": renamed, "warnings": warns, "samples": samples}
    (root / "voice2utau_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


def merge_into(target: Path, zh_bank: Path) -> int:
    """Copy a Mandarin bank's wavs and oto entries into an existing bank (in place). Returns entries added."""
    taken = read_aliases(target)
    new = (zh_bank / "oto.ini").read_bytes().decode("cp932").splitlines()
    clash = [l.split("=", 1)[1].split(",")[0] for l in new if l.split("=", 1)[1].split(",")[0] in taken]
    if clash:
        raise ValueError(f"aliases already in {target}: {clash[:10]} - rebuild with taken= so they get the '{SUFFIX}' suffix")
    for line in new:
        wav = line.split("=", 1)[0]
        if (target / wav).exists():
            raise ValueError(f"{wav} already exists in {target}")
    for line in new:
        wav = line.split("=", 1)[0]
        shutil.copy2(zh_bank / wav, target / wav)
    oto = target / "oto.ini"
    old = oto.read_bytes()
    sep = b"" if old.endswith((b"\n", b"\r\n")) or not old else b"\r\n"
    oto.write_bytes(old + sep + ("\r\n".join(new) + "\r\n").encode("cp932"))
    return len(new)


def export_dataset(banks: list[Path], zip_path: Path, min_s: float = 0.5) -> dict:
    """Zip every wav under the banks as 44.1 kHz mono 16-bit, ready for RVC training (frq/llsm files are skipped)."""
    n, total_s = 0, 0.0
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for b in banks:
            for p in sorted(b.rglob("*.wav")):
                if p.name.lower() == "sample.wav":
                    continue
                x, sr = audio.read_wav(p)
                if len(x) / sr < min_s:
                    continue
                tmp = zip_path.with_suffix(".tmp.wav")
                audio.write_wav(tmp, audio.resample(x, sr, audio.SR_BANK), audio.SR_BANK)
                zf.write(tmp, f"dataset/{b.name}_{n:05d}.wav")
                tmp.unlink()
                n += 1
                total_s += len(x) / sr
        zf.writestr("TRAINING.txt", TRAINING_TXT)
    return {"clips": n, "minutes": round(total_s / 60, 1), "zip": str(zip_path)}


TRAINING_TXT = """Training an RVC model on this dataset (e.g. with Applio or RVC WebUI)
- Put the wavs in a dataset folder; model name e.g. CASE; RVC v2, 40k sample rate, f0 method rmvpe.
- Roughly 200-400 epochs (watch the loss; stop before it overfits), batch size to fit your GPU.
- Train the index too ("Train feature index").
- Copy the final CASE.pth (not G_*.pth/D_*.pth) and CASE.index back, then run:
    python -m voice2utau.mandarin_native build --bank <bank> --rvc-model CASE.pth --rvc-index CASE.index -o out --into <bank>
Note: the clips are the character's own English/Japanese material. Check you may use the voice this way.
"""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset", help="export an RVC training set from one or more banks")
    d.add_argument("banks", nargs="+", type=Path)
    d.add_argument("-o", "--out", type=Path, required=True)
    b = sub.add_parser("build", help="make native Mandarin in the character's voice")
    b.add_argument("--bank", type=Path, help="bank to measure the pitch from (and the aliases to avoid)")
    b.add_argument("--f0", type=float, help="target pitch in Hz (default: measured from --bank)")
    b.add_argument("--rvc-model", type=Path)
    b.add_argument("--rvc-index", type=Path)
    b.add_argument("--rvc-command")
    b.add_argument("--preview-template", action="store_true", help="no RVC: raw espeak voice, for previewing only")
    b.add_argument("--only", nargs="*", help="only these pinyin syllables")
    b.add_argument("--name", default="Mandarin")
    b.add_argument("-o", "--out", type=Path, required=True)
    b.add_argument("--into", type=Path, help="also merge the result into this bank (in place)")
    a = ap.parse_args()
    if a.cmd == "dataset":
        print(export_dataset(a.banks, a.out))
        return
    if not a.bank and not a.f0:
        ap.error("give --bank (to measure the pitch) or --f0")
    f0 = a.f0 or estimate_f0(a.bank)
    backend = rvc.make_backend(a.rvc_model, a.rvc_index, a.rvc_command) if a.rvc_model else None
    taken = read_aliases(a.into or a.bank) if (a.into or a.bank) else set()
    root = build(a.out, a.name, f0, backend, allow_template=a.preview_template, taken=taken, only=a.only)
    rep = json.loads((root / "voice2utau_report.json").read_text(encoding="utf-8"))
    print(f"{rep['syllables']} syllables at {f0:.0f} Hz -> {root} (converted {rep['converted']}, "
          f"renamed {rep['renamed_aliases']}, warnings {len(rep['warnings'])})")
    if a.into:
        print(f"merged {merge_into(a.into, root)} entries into {a.into}")


if __name__ == "__main__":
    main()
