"""Sam: a fun, standalone voicebank built on Microsoft Sam (the Microsoft Speech API 4 text-to-speech voice).

Unlike the other characters this one *is* a voice on purpose - a synthetic one, not a person. Sam "records" scripted
lines (via a SAPI 4 service), and because the script is known the clips are cut by forced alignment instead of
guessing what was said:
  * Japanese CV: each line is six morae separated by pauses ("kah. kee. koo."), so the n-th sound is the n-th mora.
  * English Arpasing: each line is eight words separated by pauses; each word's CMU-dictionary pronunciation is
    aligned to the audio with the wav2vec2 phoneme model (CTC forced alignment), and every unit the word contains
    ("- C", "C V", "V C", "C C", "V V", "V -") is cut from it. The best-aligned instance of each unit wins.

    python -m voice2utau.sam japanese REC_DIR OUT
    python -m voice2utau.sam english REC_DIR OUT
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import audio
from .bank import write_bank, zip_bank
from .extract import Clip
from .morae import MORAE

SR = audio.SR_BANK
BANK_F0 = 100.0                    # Sam at SAPI pitch 100 speaks at ~99 Hz
SPELL_V = {"a": "ah", "i": "ee", "u": "oo", "e": "eh", "o": "oh"}
VOICELESS = {"k", "ky", "s", "sh", "t", "ch", "ts", "h", "hy", "f", "p", "py"}
VOICED_MS = {"n": 70, "ny": 80, "m": 70, "my": 80, "b": 30, "by": 45, "d": 30, "g": 35, "gy": 45, "r": 35, "ry": 50,
             "y": 70, "w": 70, "z": 80, "j": 75}
PER_LINE = 6


def ja_words() -> list[tuple[str, str]]:
    """(mora key, how to spell it for an English TTS) in script order."""
    out = []
    for key, m in MORAE.items():
        if not m.vowel:                                    # ん
            out.append((key, "un"))
        else:
            out.append((key, m.cls + SPELL_V[m.vowel]))
    return out


def ja_script() -> list[str]:
    w = [s for _, s in ja_words()]
    return [". ".join(w[i:i + PER_LINE]) + "." for i in range(0, len(w), PER_LINE)]


def segments(x: np.ndarray, sr: int, rel_db: float = -30.0, min_gap_ms: float = 120.0,
             min_len_ms: float = 60.0) -> list[tuple[int, int]]:
    """Sounding stretches between pauses, in samples."""
    hop = int(0.01 * sr)
    e = audio.rms_envelope(x, sr, 10.0)
    on = 20 * np.log10(e / max(1e-12, float(e.max())) + 1e-12) > rel_db
    segs: list[list[int]] = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if segs and i - segs[-1][1] < min_gap_ms / 10:
                segs[-1][1] = j
            else:
                segs.append([i, j])
            i = j
        else:
            i += 1
    return [(a * hop, b * hop) for a, b in segs if (b - a) * 10 >= min_len_ms]


def _clip(x: np.ndarray, sr: int, a: int, b: int, pad_ms: float = 15.0) -> np.ndarray:
    p = int(pad_ms / 1000 * sr)
    return x[max(0, a - p): min(len(x), b + p)]


def _finish(y: np.ndarray, sr: int) -> np.ndarray:
    y = audio.resample(y, sr, SR) if sr != SR else y
    y, _ = audio.flatten_pitch(y, SR, BANK_F0)
    return audio.fade(audio.normalise_level(y), SR)


def consonant_s(y: np.ndarray, sr: int, cls: str, pad_ms: float = 15.0) -> float:
    if not cls:
        return pad_ms / 1000
    if cls in VOICELESS:
        on = audio.voicing_onset(y, sr)
        if on is not None:
            return max(pad_ms / 1000 + 0.02, on)
        return pad_ms / 1000 + 0.08
    return pad_ms / 1000 + VOICED_MS.get(cls, 50) / 1000


def build_japanese(rec_dir: Path, out: Path, name: str = "Sam") -> Path:
    words = ja_words()
    files = sorted(Path(rec_dir).glob("*.wav"))
    clips, report = {}, {"samples": {}}
    for li, f in enumerate(files):
        x, sr = audio.read_wav(f)
        segs = segments(x, sr)
        line = words[li * PER_LINE:(li + 1) * PER_LINE]
        if len(segs) != len(line):
            report.setdefault("skipped_lines", []).append({"file": f.name, "segments": len(segs), "expected": len(line)})
            continue
        for (key, spelled), (a, b) in zip(line, segs):
            m = MORAE[key]
            y = _clip(x, sr, a, b)
            nasal = not m.vowel
            if nasal:                                          # "un": keep the nasal end
                y = y[int(len(y) * 0.45):]
            split = consonant_s(y, sr, "" if nasal else m.cls)
            clips[key] = Clip(m, _finish(y, sr), split, BANK_F0, origin=f"{f.name}:{spelled}")
            report["samples"][key] = {"source": "recorded", "origin": f"{f.name} '{spelled}'"}
    root = write_bank(clips, Path(out), name, report)
    (root / "readme.txt").write_text(
        f"{name}\nA Japanese CV voicebank built on Microsoft Sam (the Microsoft Speech API 4 text-to-speech voice): "
        "Sam spoke every mora (spelled for him, e.g. 'kah', 'shee', 'tsoo'), and each was cut by forced alignment, "
        "pitch-flattened to 100 Hz and levelled. Pure Sam - robotic on purpose.\nA fun one-off, not part of the "
        "character group. Microsoft Sam belongs to Microsoft; personal use only.\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------------------------------------------- English
ARPA_IPA = {"iy": "iː", "ih": "ɪ", "ey": "eɪ", "eh": "ɛ", "ae": "æ", "aa": "ɑː", "ao": "ɔː", "ow": "oʊ", "uh": "ʊ",
            "uw": "uː", "ah": "ʌ", "ax": "ə", "er": "ɚ", "ay": "aɪ", "aw": "aʊ", "oy": "ɔɪ", "p": "p", "b": "b", "t": "t",
            "d": "d", "k": "k", "g": "ɡ", "ch": "tʃ", "jh": "dʒ", "f": "f", "v": "v", "th": "θ", "dh": "ð", "s": "s",
            "z": "z", "sh": "ʃ", "zh": "ʒ", "hh": "h", "m": "m", "n": "n", "ng": "ŋ", "l": "l", "r": "ɹ", "w": "w",
            "y": "j"}
EN_VOWELS = {"iy", "ih", "ey", "eh", "ae", "aa", "ao", "ow", "uh", "uw", "ah", "ax", "er", "ay", "aw", "oy"}
NEAR_V = {"iy": ["ih", "ey"], "ih": ["iy", "eh"], "ey": ["eh", "iy"], "eh": ["ae", "ih"], "ae": ["eh", "aa"],
          "aa": ["ao", "ah"], "ao": ["aa", "ow"], "ow": ["ao", "uw"], "uh": ["uw", "ah"], "uw": ["uh", "ow"],
          "ah": ["ax", "aa"], "ax": ["ah", "eh"], "er": ["ah", "ax"], "ay": ["aa", "ae"], "aw": ["aa", "ow"],
          "oy": ["ow", "ao"]}


def pron(word: str) -> list[str] | None:
    import cmudict
    d = cmudict.dict().get(word.lower())
    if not d:
        return None
    return [("ax" if p == "AH0" else re.sub(r"\d", "", p).lower()) for p in d[0]]


class Aligner:
    """CTC forced alignment of a known phone string with the wav2vec2 phoneme model."""

    def __init__(self):
        from .phonemes import PhonemeRecognizer
        self.rec = PhonemeRecognizer()
        self.tok = {t: i for i, t in self.rec.id2tok.items()}

    def align(self, x16: np.ndarray, phones: list[str]) -> list[tuple[float, float, float]] | None:
        """[(start s, end s, mean prob)] per phone, or None."""
        import torch
        import torchaudio.functional as F
        from .phonemes import FRAME_S
        ids = [self.tok.get(ARPA_IPA.get(p, ""), None) for p in phones]
        if None in ids or len(x16) < audio.SR_REC // 10:
            return None
        inp = self.rec.fe(x16, sampling_rate=audio.SR_REC, return_tensors="pt").input_values
        with torch.inference_mode():
            lp = torch.log_softmax(self.rec.model(inp).logits.float(), dim=-1)
        if lp.shape[1] < len(ids) * 2:
            return None
        try:
            al, sc = F.forced_align(lp, torch.tensor([ids], dtype=torch.int32), blank=self.rec.blank)
        except Exception:
            return None
        spans = F.merge_tokens(al[0], sc[0].exp())
        if len(spans) != len(ids):
            return None
        return [(s.start * FRAME_S, s.end * FRAME_S, float(s.score)) for s in spans]


def en_units(phones: list[str]) -> list[tuple[str, str, int]]:
    """(first, second, index of the first phone; -1 for a start) for every Arpasing unit inside one word."""
    out = [("-", phones[0], -1)]
    out += [(phones[i], phones[i + 1], i) for i in range(len(phones) - 1)]
    out.append((phones[-1], "-", len(phones) - 1))
    return out


def build_english(rec_dir: Path, out: Path, name: str = "Sam_EN", lines_json: Path | None = None) -> Path:
    from .sfxbank import write_frq
    rec_dir = Path(rec_dir)
    lines = json.loads(Path(lines_json or rec_dir.parent.parent / "en_lines2.json").read_text())
    al = Aligner()
    best: dict[tuple[str, str], tuple[float, np.ndarray, float]] = {}
    skipped = []
    for li, f in enumerate(sorted(rec_dir.glob("*.wav"))):
        words = [w.strip(" .") for w in lines[li].split(".") if w.strip(" .")]
        x, sr = audio.read_wav(f)
        segs = []
        for gap in (120, 200, 300, 80):
            segs = segments(x, sr, min_gap_ms=gap)
            if len(segs) == len(words):
                break
        if len(segs) != len(words):
            skipped.append(f.name)
            continue
        for w, (a, b) in zip(words, segs):
            ph = pron(w)
            if not ph:
                continue
            pad = int(0.06 * sr)
            a0, b0 = max(0, a - pad), min(len(x), b + pad)
            seg = x[a0:b0]
            t = al.align(audio.resample(seg, sr, audio.SR_REC).astype("float32"), ph)
            if not t:
                continue
            for p, q, i in en_units(ph):
                if q == "-":                                   # ending: from the phone to the end of the word
                    s0, s1, bound = t[i][0], len(seg) / sr, t[i][1]
                    conf = t[i][2]
                elif p == "-":                                 # start: from just before the word into the phone
                    s0, bound = max(0.0, t[0][0] - 0.05), t[0][0]
                    s1 = len(seg) / sr if len(ph) == 1 else t[0][1]
                    conf = t[0][2]
                else:
                    last = i + 1 == len(ph) - 1
                    s0, bound = t[i][0], t[i + 1][0]
                    s1 = (len(seg) / sr) if last else t[i + 1][1]
                    conf = (t[i][2] + t[i + 1][2]) / 2
                if s1 - s0 < 0.06:
                    continue
                vlen = (s1 - bound) if q in EN_VOWELS else 0.0
                score = conf + 0.6 * min(vlen, 0.4)
                key = (p, q)
                if key not in best or score > best[key][0]:
                    y = seg[int(s0 * sr): int(s1 * sr)]
                    best[key] = (score, y.copy(), bound - s0)
    root = Path(out) / name
    root.mkdir(parents=True, exist_ok=True)
    oto = []
    have = {}
    for (p, q), (score, y, bound_s) in sorted(best.items()):
        y = audio.resample(y, sr, SR) if sr != SR else y
        y, _ = audio.flatten_pitch(y, SR, BANK_F0)
        y = audio.fade(audio.normalise_level(y), SR, in_ms=3.0, out_ms=8.0)
        nm = "_".join("sil" if s == "-" else s for s in (p, q))
        audio.write_wav(root / f"{nm}.wav", y.astype("float32"), SR)
        write_frq(root / f"{nm}_wav.frq", len(y), hz=BANK_F0)
        dur = len(y) / SR * 1000
        pre = max(5.0, min(bound_s * 1000, dur - 10))
        con = min(dur - 5, pre + (60 if q in EN_VOWELS else 35))
        ovl = 0.0 if p == "-" else min(pre * 0.4, 40.0)
        oto.append(f"{nm}.wav={p} {q},0,{con:.1f},{-dur:.1f},{pre:.1f},{ovl:.1f}")
        have[f"{p} {q}"] = oto[-1]
    # stand-ins for units Sam never produced: nearest vowel with the same consonant, the ending/start halves otherwise
    from .arpasing import units as arpa_units
    filled = 0
    for u in arpa_units():
        a, b = u
        al_ = f"{a} {b}"
        if al_ in have:
            continue
        subs = []
        if b in EN_VOWELS:
            subs += [f"{a} {v}" for v in NEAR_V.get(b, [])] + [f"- {b}"]
        if a in EN_VOWELS:
            subs += [f"{v} {b}" for v in NEAR_V.get(a, [])]
            if b not in EN_VOWELS:                             # "ow w", "iy y", "ay hh": English has no such coda
                subs += [f"{a} -"] + [f"{v} -" for v in NEAR_V.get(a, [])]
        if a not in EN_VOWELS and a != "-" and b not in EN_VOWELS:
            subs += [f"{a} -", f"- {b}"]
        if b == "-":
            subs += [f"{v} -" for v in NEAR_V.get(a, [])]
        if a == "-":
            subs += [f"- {v}" for v in NEAR_V.get(b, [])]
        for s in subs:
            if s in have:
                f_, rest = have[s].split("=", 1)
                oto.append(f"{f_}={al_}," + rest.split(",", 1)[1])
                filled += 1
                break
    (root / "oto.ini").write_bytes(("\r\n".join(oto) + "\r\n").encode("cp932"))
    (root / "character.txt").write_bytes(f"name={name}\r\nimage=\r\nauthor=\r\nweb=\r\n".encode("cp932"))
    (root / "character.yaml").write_text("text_file_encoding: shift_jis\nportrait_opacity: 0.67\n"
                                         "default_phonemizer: OpenUtau.Plugin.Builtin.ArpasingPhonemizer\n",
                                         encoding="utf-8")
    (root / "readme.txt").write_text(
        f"{name}\nAn English (Arpasing) voicebank built on Microsoft Sam (the Microsoft Speech API 4 text-to-speech "
        f"voice): {len(best)} units cut from Sam saying common words, by forced alignment of each word's CMU-dictionary "
        f"pronunciation; {filled} more aliases borrow the nearest unit he did say. Pure Sam, pitch-flattened to 100 Hz."
        "\nUse it in OpenUtau with the Arpasing phonemizer. A fun one-off, not part of the character group. Microsoft "
        "Sam belongs to Microsoft; personal use only.\n", encoding="utf-8")
    (root / "sam_report.json").write_text(json.dumps({"units": len(best), "filled": filled, "skipped_lines": skipped},
                                                     indent=1))
    return root


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["japanese", "english", "script"])
    ap.add_argument("rec_dir", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.kind == "script":
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "ja_lines.json").write_text(json.dumps(ja_script(), indent=0))
        return
    root = build_japanese(a.rec_dir, a.out) if a.kind == "japanese" else build_english(a.rec_dir, a.out)
    print("wrote", root, "->", zip_bank(root, root.parent / f"{root.name}.zip"))


if __name__ == "__main__":
    main()
