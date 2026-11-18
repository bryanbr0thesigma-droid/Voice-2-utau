import numpy as np
import soundfile as sf

from voice2utau import arpasing, joki, synthbank


def test_non_rhotic_r_only_after_vowels_and_before_silence():
    assert arpasing.derhotic(["aa", "r"]) == ["aa", "R0"]
    assert arpasing.derhotic(["r", "-"]) == ["R0", "-"]
    assert arpasing.derhotic(["r", "iy"]) == ["r", "iy"]          # "red", the r of "sorry": still an r
    assert arpasing.derhotic(["t", "r"]) == ["t", "r"]
    assert "R0" not in arpasing.CONS_LIST                         # no extra units in the bank


def test_recipe_accent_applies_inside_using_and_is_restored():
    before_v, before_c = dict(arpasing.VOWELS), dict(arpasing.CONS)
    r = arpasing.Recipe("x", vowels={"ow": ((560, 1450, 2800), (470, 1150, 2700))}, non_rhotic=True)
    with arpasing.using(r):
        assert arpasing.VOWELS["ow"][0][1] == 1450 and arpasing.NON_RHOTIC[0]
    assert arpasing.VOWELS == before_v and arpasing.CONS == before_c and not arpasing.NON_RHOTIC[0]


def test_british_japanese_patches_and_restores_the_cv_synth():
    v, c, g = dict(synthbank.VOWELS), dict(synthbank.CLASSES), dict(synthbank.GLIDE_START)
    with joki.british_japanese():
        assert synthbank.CLASSES["r"][0] == "glide" and synthbank.GLIDE_START["r"][2] < 2000     # [ɹ], low F3
        assert synthbank.CLASSES["t"][5]["vot"] > c["t"][5]["vot"]                                # aspirated
        assert synthbank.VOWELS["o"] != v["o"]
    assert synthbank.VOWELS == v and synthbank.CLASSES == c and synthbank.GLIDE_START == g


def test_pop_is_short_and_soft(tmp_path):
    sr = joki.SR
    (tmp_path / "desktop").mkdir()
    t = np.arange(int(0.3 * sr)) / sr
    sf.write(tmp_path / "desktop" / "pop_sound.wav", 0.7 * np.sin(2 * np.pi * 600 * t) * np.exp(-t / 0.03), sr)
    y = np.zeros(int(0.5 * sr)); c = int(0.1 * sr)
    y[c:] = 0.3 * np.sin(2 * np.pi * 200 * np.arange(len(y) - c) / sr)
    out, c2 = joki.pop(tmp_path)(y, c, None)
    nz = np.nonzero(np.abs(out - y) > 1e-9)[0]
    assert c2 == c and nz.min() >= c - int(0.008 * sr) and nz.max() < c + int(0.07 * sr)


def test_joki_is_a_young_british_man():
    r = joki.recipe.__defaults__
    assert joki.F0 > 150 and 0.88 < joki.SCALE < 0.95
    assert joki.BRITISH_EN["er"][0][2] > 2400                  # NURSE without r colour
