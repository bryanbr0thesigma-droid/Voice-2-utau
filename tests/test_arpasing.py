from pathlib import Path

import numpy as np

from voice2utau import arpasing as en


def test_unit_set_covers_arpasing():
    us = {en.alias(u) for u in en.units()}
    V, C = len(en.VOWEL_LIST), len(en.CONS_LIST)
    assert len(us) == 2 * V + 2 * C + C * V * 2 + V * V + C * (C - 1)
    for a in ["- aa", "aa -", "- s", "k -", "s t", "b iy", "iy n", "ow ah", "t r"]:
        assert a in us


def test_render_places_phones_and_keeps_vowel_level():
    rng = np.random.default_rng(0)
    with en.using(en.Recipe("T")):
        ref = en.calibrate(rng)
        for u in (["b", "aa"], ["s", "iy"], ["aa", "-"]):
            y, segs = en.render(u, rng, ref)
            assert len(y) == segs[-1].e and np.isfinite(y).all()
        y, segs = en.render(["m", "aa"], rng, ref)
        v = y[segs[1].s + 4000:segs[1].s + 12000]
        assert abs(np.sqrt((v ** 2).mean()) - ref) / ref < 0.35        # vowels land near the bank's reference level


def test_oto_fields_align_to_second_phone():
    segs = en.plan(["k", "aa"])
    con, cut, pre, ovl = en.oto_fields(["k", "aa"], segs, segs[-1].e)
    assert abs(pre - segs[1].s / en.SR * 1000) < 0.01 and con > pre and cut < 0 and 0 <= ovl < pre


def test_build_writes_bank(tmp_path):
    root = en.build(en.Recipe("T_EN"), tmp_path, only=[["-", "aa"], ["s", "aa"], ["aa", "s"], ["s", "-"]])
    lines = (root / "oto.ini").read_bytes().decode("cp932").split()
    assert len([l for l in lines if "=" in l]) == 4 and (root / "s_aa.wav").exists()
    assert "ArpasingPhonemizer" in (root / "character.yaml").read_text()
