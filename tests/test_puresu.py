import numpy as np
import soundfile as sf

from voice2utau import andro, puresu


def test_tick_is_short_faint_and_at_the_syllable_start(tmp_path):
    sr = puresu.SR
    (tmp_path / "ps3").mkdir()
    t = np.arange(int(0.08 * sr)) / sr
    sf.write(tmp_path / "ps3" / "cursor.wav", 0.8 * np.sin(2 * np.pi * 3000 * t) * np.exp(-t / 0.01), sr)
    y = np.zeros(int(0.5 * sr)); c = int(0.1 * sr)
    y[c:] = 0.3 * np.sin(2 * np.pi * 200 * np.arange(len(y) - c) / sr)
    out, c2 = puresu.tick(tmp_path, level_db=-22.0)(y, c, None)
    assert c2 == c
    nz = np.nonzero(np.abs(out - y) > 1e-9)[0]
    assert nz.min() >= c - int(0.005 * sr) and nz.max() < c + int(0.03 * sr)
    rms = np.sqrt(((out - y)[nz.min():nz.max() + 1] ** 2).mean())
    assert rms < 0.3 / np.sqrt(2) * 10 ** (-22 / 20) * 1.5                          # faint, under the vowel


def test_tick_without_the_sample_is_a_no_op(tmp_path):
    y = np.ones(1000)
    out, c = puresu.tick(tmp_path)(y, 10, None)
    assert c == 10 and np.array_equal(out, y)


def test_puresu_is_male_and_lower_than_andro():
    assert puresu.TONES and puresu.CLICKS and puresu.HISS and puresu.PAD
    assert puresu.F0 < andro.MALE_F0 and puresu.SCALE < andro.MALE_SCALE
