"""Replace the rough, English-approximated Japanese/Mandarin entries of a bank with RVC-converted clips.

"Rough" = sounds English has no counterpart for:
  Mandarin: every syllable with a retroflex initial (zh ch sh r), an ü final (ju lv xue yuan ...), or the
            buzzy vowel (zi ci si; zhi chi shi ri are covered by the retroflex rule)
  Japanese: the ら row (English has no tap; mapped to "l") and つ
Native espeak-ng templates (cmn-latn-pinyin / ja) are converted with the bank's RVC model, then flattened to the
bank pitch and level-matched. Only these entries change; if RVC returns silence for one, its old entry is kept.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import audio, rvc
from .crossling import oto_line
from .extract import Clip
from .mandarin import FINALS, syllables
from .morae import MORAE, Mora
from .templates import EspeakTemplate

JA_ROUGH = ["ra", "ri", "ru", "re", "ro", "rya", "ryu", "ryo", "tsu"]      # MORAE keys
ZH_VOICE = "cmn-latn-pinyin+f3"


def zh_rough() -> list[str]:
    out = []
    for name, (ini, fin) in syllables().items():
        if ini in ("zh", "ch", "sh", "r") or fin in ("v", "ve", "van", "vn") or name in ("zi", "ci", "si"):
            out.append(name)
    return sorted(out)


def _espeak_pitch(target_hz: float) -> int:
    return int(np.clip(60 + (target_hz - 218) / 2.57, 20, 99))      # measured: p60 -> 218 Hz, p90 -> 295 Hz


def zh_template(name: str, target_hz: float) -> Clip | None:
    sr = audio.SR_BANK
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "t.wav"
        subprocess.run(["espeak-ng", "-v", ZH_VOICE, "-s", "110", "-p", str(_espeak_pitch(target_hz)), "-w", str(w),
                        f"{name}1"], check=True, capture_output=True)       # tone 1 = flat pitch
        x, s = audio.read_wav(w)
    x = audio.resample(x, s, sr)
    a, b = audio.trim_silence(x, sr)
    x = x[a:b]
    if len(x) < sr * 0.15:
        return None
    ini, fin = syllables()[name]
    glide, _, coda = FINALS[fin]
    onset = audio.voicing_onset(x, sr)
    split = onset if onset is not None and 0.03 <= onset <= len(x) / sr * 0.7 else 0.10
    keep = split + (0.45 if (coda or glide) else 0.28)                     # keep room for the ending
    x = audio.fade(x[: int(keep * sr)], sr)
    return Clip(Mora(name, name, ini, "", "CV", "zh"), x, split, audio.median_f0(x[len(x) // 3:], sr), "template", 1.0, "espeak-cmn")


def fill(bank: Path, backend: rvc.RVCBackend) -> dict:
    rep_path = bank / "voice2utau_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    bank_f0 = float(rep["bank_f0_hz"])
    sr = audio.SR_BANK
    clips: list[Clip] = []
    ja_tpl = EspeakTemplate(bank_f0, lang="ja")
    for key in JA_ROUGH:
        c = ja_tpl.get(MORAE[key])
        if c:
            clips.append(c)
    for name in zh_rough():
        c = zh_template(name, bank_f0)
        if c:
            clips.append(c)
    f0s = [c.f0 for c in clips if c.f0]
    transpose = int(np.clip(round(12 * np.log2(bank_f0 / np.median(f0s))), -12, 12)) if f0s else 0
    out, warns = rvc.convert_clips(clips, backend, transpose)

    lines = (bank / "oto.ini").read_bytes().decode("cp932").splitlines()
    alias_of = [l.split("=", 1)[1].split(",")[0] for l in lines]
    wav_of = [l.split("=", 1)[0] for l in lines]
    result = {"transpose": transpose, "japanese": [], "mandarin": [], "kept_old_entry": warns[:], "flattened": 0}
    for c in out:
        if c.source != "rvc":                                   # RVC gave nothing usable: keep the old entry
            continue
        y, changed = audio.flatten_pitch(c.audio, sr, bank_f0)
        result["flattened"] += changed
        y = audio.normalise_level(y)
        dur = len(y) / sr * 1000
        pre = min(max(c.split_s * 1000, 15.0), dur * 0.6)
        params = [0.0, min(pre + 45, dur - 10), -dur, pre, max(5.0, pre / 3)]
        is_ja = c.mora.lang == "ja"
        wav = f"rv_ja_{c.mora.key}" if is_ja else f"rv_{c.mora.key}"
        audio.write_wav(bank / f"{wav}.wav", y, sr)
        targets = [c.mora.kana] if is_ja else [c.mora.key]
        old = next((wav_of[i] for i, a in enumerate(alias_of) if a == targets[0]), None)
        if is_ja:                                                 # the romaji alias too, if it still plays the same old file
            targets += [a for i, a in enumerate(alias_of) if a == c.mora.key and wav_of[i] == old]
        for t in targets:
            for i, a in enumerate(alias_of):
                if a == t:
                    lines[i] = oto_line(wav, t, params)
        (result["japanese"] if is_ja else result["mandarin"]).append(c.mora.kana)
    (bank / "oto.ini").write_bytes(("\n".join(lines) + "\n").encode("cp932"))
    # drop spliced files nothing refers to any more
    used = {l.split("=", 1)[0] for l in lines}
    for p in list(bank.glob("zh_*.wav")) + list(bank.glob("ja_*.wav")):
        if p.name not in used:
            p.unlink()
    rep["rvc_filled"] = result
    rep_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = bank / "readme.txt"
    readme.write_text(readme.read_text(encoding="utf-8") +
                      "RVC-filled (no English counterpart): Mandarin ü / retroflex / buzzy-vowel syllables and the Japanese ra row and tsu "
                      "(files rv_*.wav); everything else is built from the real English recordings.\n", encoding="utf-8")
    return result


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="RVC-fill the rough Japanese/Mandarin entries of a bank (in place).")
    ap.add_argument("bank", type=Path)
    ap.add_argument("--rvc-model", type=Path, required=True)
    ap.add_argument("--rvc-index", type=Path)
    ap.add_argument("--rvc-command")
    a = ap.parse_args()
    r = fill(a.bank, rvc.make_backend(a.rvc_model, a.rvc_index, a.rvc_command))
    print(f"replaced {len(r['mandarin'])} Mandarin + {len(r['japanese'])} Japanese entries; "
          f"transpose {r['transpose']:+d}; flattened {r['flattened']}; kept old entry for {len(r['kept_old_entry'])}")


if __name__ == "__main__":
    main()
