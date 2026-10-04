#!/usr/bin/env python
"""Build voice2utau/data/en_unit_freq.json: how often each English bank unit occurs in running English.

usage: python tools/build_unit_freq.py texts.txt        (one sentence per line)
The shipped table was built from the public-domain LJSpeech transcripts (~1,600 sentences).
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from voice2utau import english
from voice2utau.phonemes import Phone

TOK = re.compile(r"tʃ|dʒ|aɪ|eɪ|oʊ|aʊ|ɔɪ|əʊ|[^\sˈˌ.,;:!?\-–—'\"()_]ː?[̀-ͯ]*")


def units_of(text: str) -> list[str]:
    ipa = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text], capture_output=True, text=True).stdout
    toks = [t for t in TOK.findall(ipa.replace("\n", " ").replace("_", " ")) if english.classify(t).kind in ("V", "C")]
    phones = [Phone(t, i * 0.06, i * 0.06 + 0.04, 1.0) for i, t in enumerate(toks)]
    return [c.mora.key for c in english.build_candidates(phones, "x")]


def main(path: str) -> None:
    counts: Counter = Counter()
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for l in lines:
        counts.update(units_of(l))
    total = sum(counts.values())
    table = {k: round(v / total, 7) for k, v in counts.most_common()}
    out = Path(__file__).resolve().parent.parent / "voice2utau" / "data" / "en_unit_freq.json"
    out.write_text(json.dumps({"source": f"{len(lines)} sentences of public-domain English text (LJSpeech transcripts)",
                               "units": table}, indent=0), encoding="utf-8")
    print(f"{len(lines)} sentences, {total} unit tokens, {len(table)} distinct units -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
