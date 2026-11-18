import numpy as np

from voice2utau import characters, english_recipes, synthbank


def _centroid(y, sr):
    X = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / sr)
    return float((X * f).sum() / X.sum())


def test_mature_source_is_warmer_and_breathier_than_the_plain_voice():
    sr = synthbank.SR
    with characters.mado_voice():
        plain = synthbank.glottal(sr, np.random.default_rng(0))
        mature = characters.mature_source()(sr, np.random.default_rng(0))
        mature_dry = characters.mature_source(breath=0.0)(sr, np.random.default_rng(0))
    assert np.abs(mature).max() <= 1.0 + 1e-9
    assert _centroid(mature_dry, sr) < _centroid(plain, sr)                 # darker tilt
    hi = lambda y: np.abs(np.fft.rfft(y))[int(4000 * len(y) / sr):].mean() / np.abs(np.fft.rfft(y)).mean()
    assert hi(mature) > hi(mature_dry)                                     # breath fills the top end


def test_mado_voice_settings_restore_and_recipe_uses_them():
    f0, sc = synthbank.F0, synthbank.FORMANT_SCALE
    with characters.mado_voice():
        assert synthbank.F0 == characters.MADO_F0 < 261.63 and synthbank.FORMANT_SCALE < 1.0
    assert (synthbank.F0, synthbank.FORMANT_SCALE) == (f0, sc)
    r = english_recipes.mado()
    assert r.f0 == characters.MADO_F0 and r.scale == characters.MADO_SCALE and r.source is not None
