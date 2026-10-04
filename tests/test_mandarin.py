import json

from tests.test_crossling import make_bank
from voice2utau import audio, crossling, mandarin


def test_pinyin_table_is_complete_and_unique():
    s = mandarin.syllables()
    assert 400 <= len(s) <= 420 and len(set(s)) == len(s)
    for name in ("ma", "shi", "zhuang", "lv", "nve", "xu", "yuan", "weng", "er", "zi", "ri", "jiong", "dui", "wo"):
        assert name in s, name
    assert "ju" in s and s["ju"] == ("j", "v") and "lü" not in s            # ju = j + ü, spelled with u
    for ini, fin in s.values():                                              # every final is decomposable
        assert fin in mandarin.FINALS


def test_direct_spliced_and_collisions(tmp_path):
    keys = {"cv_m_aa": "recorded", "cv_jh_uw": "recorded", "cv_w_aa": "recorded", "vc_aa_ng": "recorded",
            "cv_t_ih": "recorded", "cv_s_ih": "rvc", "cv_sh_er": "recorded", "cv_sh_iy": "recorded", "v_aa": "recorded"}
    bank = make_bank(tmp_path, keys)
    crossling.add_japanese(bank, tmp_path / "out", "VLGR")            # adds Japanese incl. romaji "shi" -> cv_sh_iy
    root = tmp_path / "out" / "VLGR"
    s = mandarin.add_mandarin(root)
    lines = (root / "oto.ini").read_bytes().decode("cp932").splitlines()
    by = {l.split("=", 1)[1].split(",")[0]: l for l in lines}
    assert by["ma"].startswith("cv_m_aa.wav=ma,0,")                      # plain unit: alias of the existing wav
    for name in ("zhuang", "zi"):                                        # spliced: new file, referenced WITH .wav
        wav = by[name].split("=")[0]
        assert wav == f"zh_{name}.wav" and (root / wav).exists()
    assert by["shi"].startswith("cv_sh_er.wav=shi,")                     # pinyin shi (apical vowel) beats Japanese romaji shi
    assert "shi" in s["romaji_replaced_by_pinyin"]
    assert by["し"].startswith("cv_sh_iy.wav=し,")                       # ... and the hiragana name still plays the Japanese sound
    assert "ba" in s["missing"] and s["spliced"] >= 2
    dur = audio.duration(root / "zh_zhuang.wav") * 1000
    p = float(by["zhuang"].split(",")[4])
    assert 15 <= p <= dur * 0.6
    rep = json.loads((root / "voice2utau_report.json").read_text(encoding="utf-8"))
    assert rep["mandarin"]["syllables"] == s["syllables"] and "pinyin" in (root / "readme.txt").read_text()


def test_build_preutterance_counts_consonant_pieces(tmp_path):
    bank = make_bank(tmp_path, {"cv_t_ih": "recorded", "cv_s_ih": "recorded"})
    m = crossling.Mapper(bank)
    y, p = m.build([("cv_t_ih", "cons"), ("cv_s_ih", "full")])
    dur = len(y) / audio.SR_BANK * 1000
    assert p[0] == 0 and p[3] > 80 and p[3] <= dur * 0.6 and p[2] == -dur
