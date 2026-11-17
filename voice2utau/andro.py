"""ANDRO: a male Japanese CV voicebank built from the sounds of Android devices. One guy, many brands.

Core voice (no suffix): a blend of every brand. Each brand's voice tone, consonant noise and click releases are pooled,
so the default ANDRO is all of Android at once. Brand colours (alias suffix, e.g. あ_sam, ka_pix): the same singer built
only from that brand's own sounds. Plus every brand's system sounds as one-shots (<brand>_<name>).
No recorded human voice is used: every clip goes through a voice screen (see `screen`) before anything is built.

Same synthesiser as MADO, set to a male voice (lower pitch, longer vocal tract: every formant x FORMANT_SCALE).

    python -m voice2utau.andro build STAGE_DIR OUT [--exclude excluded.tsv] [--brands sam pix ...]

STAGE_DIR holds one folder per brand suffix (sam/, pix/, ...), each with that brand's mono .wav system sounds.
"""
from __future__ import annotations

import contextlib
import shutil
from pathlib import Path

import numpy as np

from . import audio, ninchan, synthbank
from .characters import _hooks

SR = synthbank.SR
MALE_F0 = 130.81                  # C3
MALE_SCALE = 0.86                 # vocal tract ~16 % longer than MADO's

# suffix -> display name. The suffix is the folder name in STAGE_DIR and the alias suffix in the bank.
BRANDS = {
    "and": "stock Android (AOSP, Nexus, LineageOS)", "pix": "Google Pixel", "sam": "Samsung", "moto": "Motorola",
    "htc": "HTC", "sony": "Sony Xperia", "lg": "LG", "hw": "Huawei", "mi": "Xiaomi", "op": "OnePlus", "oppo": "OPPO",
    "vivo": "vivo", "nok": "Nokia", "asus": "Asus", "tcl": "TCL / Alcatel", "amz": "Amazon (Fire tablets, Fire TV)",
}
# a clip whose name says it is a voice is out even if the classifier misses it (whole words, so "android" and
# "crossing" don't count)
VOICE_WORDS = {"voice", "speech", "speak", "talk", "say", "says", "droid", "bixby", "alexa", "assistant", "tts", "beatbox",
               "vocal", "vocals", "vocalise", "vocalize", "singing", "sings", "laugh", "laughing", "narrator", "announcer",
               "announcement", "countdown"}
VOICE_PHRASES = ("hello_moto", "beat_box")


@contextlib.contextmanager
def male(f0: float = MALE_F0, scale: float = MALE_SCALE):
    old = synthbank.F0, synthbank.FORMANT_SCALE
    synthbank.F0, synthbank.FORMANT_SCALE = f0, scale
    try:
        yield
    finally:
        synthbank.F0, synthbank.FORMANT_SCALE = old


def named_like_voice(p: Path) -> bool:
    import re
    s = re.sub(r"[^a-z0-9]+", "_", p.stem.lower())
    return bool(VOICE_WORDS & set(s.split("_"))) or any(ph in s for ph in VOICE_PHRASES)


def read_excluded(tsv: Path | None) -> set[str]:
    """Lines of '<prob>\\t<label>\\t<top label>\\t<path>' from the voice screen; paths listed there are excluded."""
    if not tsv or not Path(tsv).exists():
        return set()
    return {l.rstrip("\n").split("\t")[-1] for l in Path(tsv).read_text(encoding="utf-8").splitlines() if l.strip()}


def _trimmed(p: Path) -> np.ndarray:
    x, sr = audio.read_wav(p)
    x = audio.resample(x, sr, SR) if sr != SR else x
    a, b = audio.trim_silence(x, SR, rel_db=-40)
    return x[a:b]


def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())


def stock_names(and_dir: Path) -> set[str]:
    """Normalised names of the stock Android (AOSP) sounds, which many brands ship unchanged."""
    out = set()
    for p in Path(and_dir).glob("aosp*.wav"):
        n = _norm(p.stem.split("_", 2)[-1] if p.stem.startswith("aosp_material_") else p.stem.split("_", 1)[-1])
        if len(n) >= 6:
            out.add(n)
    return out


def is_stock(p: Path, stock: set[str]) -> bool:
    """A brand file that is really a stock Android sound (name after the device prefix starts with an AOSP name)."""
    n = _norm(p.stem.split("_", 1)[-1])
    return any(n.startswith(s) for s in stock)


def spread(files: list[Path], n: int) -> list[Path]:
    """Up to n files evenly across the list (not just the first n alphabetically)."""
    if len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in sorted(set(idx))]


def roles(files: list[Path]) -> dict[str, list[Path]]:
    """Sort a brand's sounds: clicks (short, sharp: become k/t/p releases), noisy (become consonant hiss), and the
    rest (voice tone candidates)."""
    import parselmouth
    clicks, noisy, tonal = [], [], []
    for p in files:
        x = _trimmed(p)
        if len(x) < int(0.01 * SR):
            continue
        dur = len(x) / SR
        env = np.abs(x)
        rise = int(np.argmax(env > 0.5 * env.max())) / SR
        if dur <= 0.3 and rise <= 0.012:
            clicks.append(p)
            continue
        seg = x[:int(3 * SR)].astype("float64")
        h = np.array(parselmouth.Sound(seg, SR).to_harmonicity_cc(0.01, 80).values[0])
        h = h[h > -100]
        (noisy if (np.median(h) if len(h) else -10) <= 3.0 else tonal).append(p)
    return {"clicks": clicks, "noisy": noisy, "tonal": tonal}


def voice_parts(r: dict[str, list[Path]], top: int = 3, keep: float = 0.4):
    """(source, noise, burst, carriers) hooks for one brand (or for the pooled core)."""
    carriers = ninchan.pick_carriers([], top=top, keep=keep, files=spread(r["tonal"], 300))
    if not carriers:
        return None
    source = ninchan.make_source([c[3] for c in carriers])
    noise = ninchan.make_noise(spread(r["noisy"], 40), mix=0.6) if r["noisy"] else None
    burst = ninchan.make_burst(spread(r["clicks"], 40)) if r["clicks"] else None
    return source, noise, burst, carriers


def merge_set(core: Path, part: Path, suffix: str) -> int:
    """Copy a brand set into the core bank as <key>_<suffix>.wav with aliases + '_<suffix>'."""
    lines = [l for l in (part / "oto.ini").read_bytes().decode("cp932").splitlines() if "=" in l]
    out, done = [], set()
    for l in lines:
        wav, rest = l.split("=", 1)
        alias, fields = rest.split(",", 1)
        stem = wav[:-4]
        new = f"{stem}_{suffix}"
        if stem not in done:
            shutil.copy2(part / wav, core / f"{new}.wav")
            frq = part / f"{stem}_wav.frq"
            if frq.exists():
                shutil.copy2(frq, core / f"{new}_wav.frq")
            done.add(stem)
        out.append(f"{new}.wav={alias}_{suffix},{fields}")
    oto = core / "oto.ini"
    have = [l for l in oto.read_bytes().decode("cp932").splitlines() if "=" in l]
    oto.write_bytes(("\n".join(have + out) + "\n").encode("cp932"))
    return len(done)


def add_oneshots(root: Path, sfx: Path | None, max_mb: float = 380.0) -> dict[str, list[str]]:
    """Put the one-shots in sub-folders (oneshots_1, oneshots_2, ...) each with its own oto.ini (UTAU reads oto.ini in
    sub-folders). Whole brands are packed into folders of at most ~max_mb, so each can ship as its own download (the
    full set is ~1 GB) and all of them unzip into the same bank folder. Returns {folder: [brands]}."""
    if not sfx or not (Path(sfx) / "oto.ini").exists():
        return {}
    sfx = Path(sfx)
    lines = [l for l in (sfx / "oto.ini").read_bytes().decode("cp932").splitlines() if "=" in l]
    by_brand: dict[str, list[str]] = {}
    for l in lines:
        by_brand.setdefault(l.split("=", 1)[0].split("_", 1)[0], []).append(l)
    size = {b: sum((sfx / w).stat().st_size for w in {l.split("=", 1)[0] for l in ls}) for b, ls in by_brand.items()}
    folders: list[list[str]] = []
    for b in sorted(size, key=lambda k: -size[k]):                 # largest brands first, first folder that fits
        for f in folders:
            if sum(size[x] for x in f) + size[b] <= max_mb * 1e6:
                f.append(b)
                break
        else:
            folders.append([b])
    for old in root.glob("oneshots*"):
        shutil.rmtree(old, ignore_errors=True)
    out = {}
    for i, f in enumerate(folders, 1):
        dst = root / f"oneshots_{i}"
        dst.mkdir(parents=True)
        keep = [l for b in f for l in by_brand[b]]
        for w in {l.split("=", 1)[0] for l in keep}:
            shutil.copy2(sfx / w, dst / w)
            frq = sfx / (w[:-4] + "_wav.frq")
            if frq.exists():
                shutil.copy2(frq, dst / frq.name)
        (dst / "oto.ini").write_bytes(("\r\n".join(keep) + "\r\n").encode("cp932", "replace"))
        out[dst.name] = sorted(f)
    return out


def build(stage: Path, out: Path, name: str = "ANDRO", brands: list[str] | None = None, exclude: set[str] = frozenset(),
          sfx_bank: Path | None = None, keep: float = 0.4) -> tuple[Path, dict]:
    stage = Path(stage)
    brands = [b for b in (brands or list(BRANDS)) if (stage / b).is_dir()]
    files = {b: sorted(p for p in (stage / b).rglob("*.wav") if str(p) not in exclude and not named_like_voice(p))
             for b in brands}
    stock = stock_names(stage / "and") if (stage / "and").is_dir() else set()
    own = {b: (files[b] if b == "and" else [p for p in files[b] if not is_stock(p, stock)]) for b in brands}
    own = {b: (own[b] or files[b]) for b in brands}           # a brand with only stock sounds keeps them
    rl = {b: roles(own[b]) for b in brands}
    report = {"brands": {}, "core": {}}
    tmp = out / "_parts"
    with male():
        # the core: every brand's tone, hiss and clicks pooled
        pooled = {"clicks": [], "noisy": [], "tonal": []}
        tones = []
        for b in brands:
            for k in pooled:
                pooled[k] += spread(rl[b][k], 12)
            best = ninchan.pick_carriers([], top=1, keep=keep, files=spread(rl[b]["tonal"], 300))
            if best:
                tones.append(best[0][3])
                report["core"][b] = Path(best[0][0]).name
        if not tones:
            raise ValueError("no tonal sound found in any brand")
        noise = ninchan.make_noise(pooled["noisy"], mix=0.6) if pooled["noisy"] else None
        burst = ninchan.make_burst(pooled["clicks"]) if pooled["clicks"] else None
        with _hooks(ninchan.make_source(tones), noise, burst, None):
            root = synthbank.build(out, name)
        for b in brands:
            parts = voice_parts(rl[b], keep=keep)
            n_files = {k: len(v) for k, v in rl[b].items()}
            n_files["stock_skipped"] = len(files[b]) - len(own[b])
            if parts is None:
                report["brands"][b] = {"files": n_files, "built": False}
                continue
            src, nz, bu, carriers = parts
            with _hooks(src, nz, bu, None):
                part = synthbank.build(tmp, f"{name}_{b}")
            n = merge_set(root, part, b)
            report["brands"][b] = {"files": n_files, "built": True, "samples": n,
                                   "tone": [Path(c[0]).name for c in carriers]}
    shutil.rmtree(tmp, ignore_errors=True)
    report["one_shot_folders"] = add_oneshots(root, sfx_bank)
    report["one_shots"] = sum(1 for d in report["one_shot_folders"]
                              for l in (root / d / "oto.ini").read_bytes().decode("cp932").splitlines() if "=" in l)
    built = [b for b in brands if report["brands"][b].get("built")]
    lines = "\n".join(f"  _{b:5s} {BRANDS[b]}" for b in built)
    (root / "readme.txt").write_text(
        f"{name}\nA male Japanese CV voicebank built from the sounds of Android devices: one guy, many brands.\n\n"
        f"Default voice (no suffix): a blend of every brand - each brand's tone, hiss and clicks pooled into one singer.\n"
        f"Brand colours: add the suffix to any lyric (あ_sam, ka_pix, 息_moto ...) to sing as that brand alone:\n{lines}\n\n"
        f"One-shots: {report['one_shots']} system sounds (ringtones, notifications, UI, startup) as <brand>_<name>, "
        "ringtones cut to 10 s, in sub-folders that each come as their own download: "
        + "; ".join(f"{d}: {', '.join(bs)}" for d, bs in report["one_shot_folders"].items())
        + ". Unzip them into this folder.\n"
        "No recorded human voice is used anywhere: every clip was screened by an audio classifier and by name.\n"
        f"Pitch: samples at C3 ({MALE_F0:.2f} Hz). Same synthesiser as MADO, with a male vocal tract.\n"
        "Render with voice2utau.sing --engine psola to keep each brand's timbre.\n"
        "The source sounds belong to Google, Samsung, Motorola, and the other device makers. Personal use only.\n",
        encoding="utf-8")
    return root, report


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("stage", type=Path)
    b.add_argument("out", type=Path)
    b.add_argument("--name", default="ANDRO")
    b.add_argument("--brands", nargs="*")
    b.add_argument("--exclude", type=Path, help="voice-screen hits to leave out")
    b.add_argument("--sfx-bank", type=Path)
    b.add_argument("--keep", type=float, default=0.4)
    a = ap.parse_args()
    root, rep = build(a.stage, a.out, a.name, a.brands, read_excluded(a.exclude), a.sfx_bank, a.keep)
    (root / "andro_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {root}: brands {[k for k, v in rep['brands'].items() if v.get('built')]}, one-shots {rep['one_shots']}")


if __name__ == "__main__":
    main()
