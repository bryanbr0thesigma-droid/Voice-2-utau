import numpy as np

from voice2utau import rigo


def test_haptic_tap_is_short_low_and_under_the_vowel():
    sr = rigo.SR
    y = np.zeros(int(0.5 * sr)); c = int(0.1 * sr)
    y[c:] = 0.3 * np.sin(2 * np.pi * 300 * np.arange(len(y) - c) / sr)
    out, c2 = rigo.haptic(level_db=-20.0)(y, c, None)
    assert c2 == c
    added = out - y
    nz = np.nonzero(np.abs(added) > 1e-9)[0]
    assert nz.min() >= c - int(0.004 * sr) and nz.max() < c + int(0.02 * sr)        # ~16 ms, at the syllable start
    assert np.abs(added).max() < 0.3 * 10 ** (-20 / 20) * 1.5                        # well under the vowel
    X = np.abs(np.fft.rfft(added)); f = np.fft.rfftfreq(len(added), 1 / sr)
    assert f[np.argmax(X)] < 400                                                     # a low thump, not a click


def test_rigo_sources_exist():
    assert rigo.TONES and rigo.CLICKS and rigo.HISS
    assert rigo.F0 < 261.63                                                          # a touch lower than MADO
