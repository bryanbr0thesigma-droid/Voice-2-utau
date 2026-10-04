"""Add Japanese (hiragana + romaji) aliases to a finished English CVVC bank, without new recordings or RVC.

Each Japanese mora is pointed at the closest existing English unit (か -> "k aa", し -> "sh iy" ...).
Sounds that have no English unit are spliced from existing clips:
    きゃ = "k" burst of cv_k_iy + cv_y_aa      つ = "t" burst of cv_t_uw + cv_s_uw      ん = nasal tail of vc_ah_n
The result keeps the English accent (his vowels are English); the mapping tables below are the knobs to tune.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio
from .morae import MORAE, Mora

# Japanese vowel / consonant -> English candidates, best first. Recorded clips are preferred over RVC ones.
VOWEL = {"a": ["aa", "ah", "ae"], "i": ["iy", "ih"], "u": ["uw", "uh"], "e": ["eh", "ey"], "o": ["ao", "ow"]}
CONS = {"k": ["k"], "g": ["g"], "s": ["s"], "sh": ["sh"], "z": ["z"], "j": ["jh"], "t": ["t"], "ch": ["ch"],
        "d": ["d"], "n": ["n"], "h": ["hh"], "f": ["f"], "b": ["b"], "p": ["p"], "m": ["m"], "y": ["y"],
        "w": ["w"], "r": ["l", "d", "r"]}
# palatalised rows (kya ...): consonant part of "<c> iy" + the full "ya/yu/yo" clip
PALATAL = {"ky": "k", "gy": "g", "ny": "n", "hy": "h", "by": "b", "py": "p", "my": "m", "ry": "r"}
XFADE_MS = 8.0


@dataclass
class Entry:
    wav: str
    alias: str
    params: list[float]       # consonant, cutoff, pre, overlap (offset is always 0)


def read_oto(path: Path) -> dict[str, tuple[str, list[float]]]:
    """wav stem -> (alias, [offset, consonant, cutoff, pre, overlap])"""
    out = {}
    for line in path.read_bytes().decode("cp932").splitlines():
        if "=" not in line:
            continue
        wav, rest = line.split("=", 1)
        f = rest.split(",")
        out[wav[:-4]] = (f[0], [float(v) for v in f[1:6]])
    return out


def _fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def oto_line(wav: str, alias: str, p: list[float]) -> str:
    return f"{wav}.wav={alias},{','.join(_fmt(v) for v in p)}"


class Mapper:
    def __init__(self, bank: Path):
        self.bank = bank
        self.oto = read_oto(bank / "oto.ini")
        rep = json.loads((bank / "voice2utau_report.json").read_text(encoding="utf-8"))
        self.recorded = {s["key"] for s in rep["samples"] if s["source"] == "recorded"}
        self.have = set(self.oto)

    def pick(self, consonants: list[str], vowels: list[str], kind: str = "cv") -> str | None:
        """First existing English unit key, preferring real recordings."""
        keys = [f"{kind}_{c}_{v}" if c else f"v_{v}" for c in consonants for v in vowels]
        keys = [k for k in keys if k in self.have]
        return next((k for k in keys if k in self.recorded), keys[0] if keys else None)

    def wav(self, key: str) -> tuple[np.ndarray, int]:
        return audio.read_wav(self.bank / f"{key}.wav")

    def splice(self, head_key: str, tail_key: str) -> tuple[np.ndarray, list[float]]:
        """consonant part of head + the whole tail clip, crossfaded. Returns (audio, oto params)."""
        hx, sr = self.wav(head_key)
        tx, _ = self.wav(tail_key)
        h_pre = self.oto[head_key][1][3]            # preutterance = end of the consonant, ms
        t_pre = self.oto[tail_key][1][3]
        head = hx[: max(int(sr * 0.02), int(sr * h_pre / 1000))]
        n = int(sr * XFADE_MS / 1000)
        if len(head) > n and len(tx) > n:
            fade = np.linspace(0, 1, n, dtype=np.float32)
            mid = head[-n:] * (1 - fade) + tx[:n] * fade
            y = np.concatenate([head[:-n], mid, tx[n:]])
        else:
            y = np.concatenate([head, tx])
        head_ms = len(head) / sr * 1000 - XFADE_MS
        pre = head_ms + t_pre                         # everything before the vowel
        dur = len(y) / sr * 1000
        pre = min(pre, dur * 0.6)
        return y, [0.0, min(pre + 45, dur - 10), -dur, pre, max(5.0, pre / 3)]

    def build(self, spec: list[tuple[str, str]]) -> tuple[np.ndarray, list[float]]:
        """Join pieces of existing clips. spec = [(unit key, mode)], mode: 'cons' (up to the vowel onset),
        'full', or 'tail' (from just before the consonant that follows a vowel). Pieces are crossfaded;
        preutterance = every 'cons' piece + the consonant length of the first 'full' piece."""
        segs, pre, seen_full = [], 0.0, False
        for key, mode in spec:
            x, sr = self.wav(key)
            p = self.oto[key][1][3]                          # preutterance of that clip, ms
            if mode == "cons":
                seg = x[: max(int(sr * 0.015), int(sr * p / 1000))]
                pre += len(seg) / sr * 1000 - XFADE_MS
            elif mode == "tail":
                seg = x[max(0, int(sr * (p - 20) / 1000)):]
            else:
                seg = x
                if not seen_full:
                    pre += p
                    seen_full = True
            segs.append(seg)
        n = int(sr * XFADE_MS / 1000)
        y = segs[0]
        for seg in segs[1:]:
            if len(y) > n and len(seg) > n:
                fade = np.linspace(0, 1, n, dtype=np.float32)
                y = np.concatenate([y[:-n], y[-n:] * (1 - fade) + seg[:n] * fade, seg[n:]])
            else:
                y = np.concatenate([y, seg])
        dur = len(y) / sr * 1000
        pre = max(15.0, min(pre, dur * 0.6))
        return y, [0.0, min(pre + 45, dur - 10), -dur, pre, max(5.0, pre / 3)]

    def nasal_tail(self, key: str) -> tuple[np.ndarray, list[float]]:
        """ん: the nasal part of a vowel+n unit."""
        x, sr = self.wav(key)
        pre = self.oto[key][1][3]                     # vowel length = where the nasal starts, ms
        y = x[max(0, int(sr * (pre - 15) / 1000)):]
        dur = len(y) / sr * 1000
        return y, [0.0, dur - 10, -dur, 15.0, 5.0]


def add_japanese(bank: Path, out: Path, name: str = "VLGR", romaji: bool = True) -> dict:
    """Copy `bank` to `out/<name>` and add the Japanese aliases. Returns a summary."""
    m = Mapper(bank)
    root = out / name
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(bank, root)
    lines: list[str] = []
    summary = {"direct": 0, "spliced": 0, "missing": [], "recorded_backed": 0, "rvc_backed": 0, "map": {}}

    def alias_lines(mora: Mora, wav: str, params: list[float]) -> None:
        lines.append(oto_line(wav, mora.kana, params))
        if romaji:
            lines.append(oto_line(wav, mora.key, params))

    def direct(mora: Mora, key: str | None) -> bool:
        if not key:
            return False
        alias_lines(mora, key, m.oto[key][1])
        summary["direct"] += 1
        summary["map"][mora.kana] = key
        summary["recorded_backed" if key in m.recorded else "rvc_backed"] += 1
        return True

    def spliced(mora: Mora, head: str | None, tail: str | None, label: str) -> bool:
        if not (head and tail):
            return False
        y, p = m.splice(head, tail)
        wav = f"ja_{mora.key}"
        audio.write_wav(root / f"{wav}.wav", y, audio.SR_BANK)
        alias_lines(mora, wav, p)
        summary["spliced"] += 1
        summary["map"][mora.kana] = f"{label}: {head} + {tail}"
        both = head in m.recorded and tail in m.recorded
        summary["recorded_backed" if both else "rvc_backed"] += 1
        return True

    for mora in MORAE.values():
        if mora.key == "nn":
            key = next((k for k in ("vc_ah_n", "vc_aa_n", "vc_ih_n") if k in m.have), None)
            if key:
                y, p = m.nasal_tail(key)
                audio.write_wav(root / "ja_nn.wav", y, audio.SR_BANK)
                alias_lines(mora, "ja_nn", p)
                summary["spliced"] += 1
                summary["map"][mora.kana] = f"nasal tail of {key}"
                summary["recorded_backed" if key in m.recorded else "rvc_backed"] += 1
                continue
        elif not mora.cls:                                           # あ い う え お
            if direct(mora, m.pick([""], VOWEL[mora.vowel])):
                continue
        elif mora.cls in PALATAL and mora.vowel == "i":              # ひ に み り き ...: plain "<c> iy"
            if direct(mora, m.pick(CONS[PALATAL[mora.cls]], VOWEL["i"])):
                continue
        elif mora.cls in PALATAL:                                    # きゃ ... : "k" of "ki" + "ya"
            base = CONS[PALATAL[mora.cls]]
            head = m.pick(base, VOWEL["i"])
            tail = m.pick(["y"], VOWEL[mora.vowel])
            if spliced(mora, head, tail, "splice"):
                continue
        elif mora.cls == "ts":                                       # つ: "t" burst + "su"
            if spliced(mora, m.pick(["t"], VOWEL["u"]), m.pick(["s"], VOWEL["u"]), "splice"):
                continue
        elif mora.cls in CONS:
            if direct(mora, m.pick(CONS[mora.cls], VOWEL[mora.vowel])):
                continue
        summary["missing"].append(mora.kana)

    old = (root / "oto.ini").read_bytes().decode("cp932").splitlines()
    (root / "oto.ini").write_bytes(("\n".join(old + lines) + "\n").encode("cp932"))
    (root / "character.txt").write_bytes(f"name={name}\nimage=\nauthor=\nweb=\n".encode("cp932"))
    rep = json.loads((root / "voice2utau_report.json").read_text(encoding="utf-8"))
    rep["name"] = name
    rep["japanese"] = {k: v for k, v in summary.items() if k != "map"} | {"aliases": "hiragana" + (" + romaji" if romaji else "")}
    rep["japanese_map"] = summary["map"]
    (root / "voice2utau_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "readme.txt").write_text(
        f"{name}\nGenerated by Voice-2-UTAU.\n"
        "English CVVC aliases (ARPAbet): CV 'k ae', VC 'ae t', initial vowel '- ae'.\n"
        "Japanese aliases (hiragana" + (" and romaji" if romaji else "") + ") reuse the English recordings, so Japanese "
        "carries the English accent; a few sounds (きゃ-row, つ, ん) are spliced from existing clips (files ja_*.wav).\n"
        "voice2utau_report.json lists which samples are recordings and which are RVC fills.\n"
        "Check the terms of the source voice and of any RVC model used before distributing.\n", encoding="utf-8")
    return summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Add Japanese aliases to an English CVVC bank.")
    ap.add_argument("bank", type=Path, help="finished bank folder (with oto.ini and voice2utau_report.json)")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"))
    ap.add_argument("--name", default="VLGR")
    ap.add_argument("--no-romaji", action="store_true")
    a = ap.parse_args()
    s = add_japanese(a.bank, a.out, a.name, not a.no_romaji)
    print(f"{len(MORAE)} Japanese sounds: {s['direct']} direct aliases, {s['spliced']} spliced, "
          f"{len(s['missing'])} missing {s['missing']}; backed by recordings: {s['recorded_backed']}, by RVC: {s['rvc_backed']}")


if __name__ == "__main__":
    main()
