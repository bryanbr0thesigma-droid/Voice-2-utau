import numpy as np
import pytest

from voice2utau import sam
from voice2utau.morae import MORAE


def test_japanese_script_covers_every_mora_in_lines_of_six():
    words = sam.ja_words()
    assert [k for k, _ in words] == list(MORAE)
    lines = sam.ja_script()
    assert sum(len(l.split(". ")) for l in lines) == len(MORAE)
    assert dict(words)["nn"] == "un" and dict(words)["ka"] == "kah" and dict(words)["shi"] == "shee"


def test_segments_split_at_pauses():
    sr = 22050
    t = np.arange(int(0.3 * sr)) / sr
    tone = 0.5 * np.sin(2 * np.pi * 120 * t)
    gap = np.zeros(int(0.4 * sr))
    x = np.concatenate([gap, tone, gap, tone, gap, tone, gap])
    segs = sam.segments(x, sr)
    assert len(segs) == 3
    assert all(abs((b - a) / sr - 0.3) < 0.03 for a, b in segs)


def test_units_inside_a_word():
    assert sam.en_units(["k", "ae", "t"]) == [("-", "k", -1), ("k", "ae", 0), ("ae", "t", 1), ("t", "-", 2)]


def test_pronunciations_use_ax_for_unstressed_schwa():
    pytest.importorskip("cmudict")
    assert sam.pron("about")[0] == "ax"
    assert all(p in sam.ARPA_IPA for p in sam.pron("question"))
