import numpy as np
import soundfile as sf

from voice2utau import andro, linx, puresu, unix


def test_drum_tap_is_short_soft_and_at_the_syllable_start(tmp_path):
    sr = linx.SR
    (tmp_path / "ubuntu").mkdir()
    t = np.arange(int(0.5 * sr)) / sr
    drum = np.zeros_like(t); drum[int(0.1 * sr):] = np.sin(2 * np.pi * 120 * t[:len(t) - int(0.1 * sr)]) * np.exp(-t[:len(t) - int(0.1 * sr)] / 0.05)
    sf.write(tmp_path / "ubuntu" / "desktop-login.wav", 0.8 * drum, sr)
    y = np.zeros(int(0.5 * sr)); c = int(0.1 * sr)
    y[c:] = 0.3 * np.sin(2 * np.pi * 200 * np.arange(len(y) - c) / sr)
    out, c2 = linx.drum_tap(tmp_path, level_db=-20.0)(y, c, None)
    nz = np.nonzero(np.abs(out - y) > 1e-9)[0]
    assert c2 == c and nz.min() >= c - int(0.009 * sr) and nz.max() < c + int(0.08 * sr)
    assert np.sqrt(((out - y)[nz.min():nz.max() + 1] ** 2).mean()) < 0.3 / np.sqrt(2) * 10 ** (-20 / 20) * 1.5


def test_warm_layer_removes_the_top_end():
    sr = linx.SR
    noise = lambda n, rng: rng.standard_normal(n)
    y = linx._warm(noise)(sr, np.random.default_rng(0))
    X = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / sr)
    assert X[f > 8000].mean() < 0.05 * X[f < 2000].mean()


def test_linx_is_a_young_man_and_skips_android_quest_steam():
    assert linx.F0 not in (andro.MALE_F0, puresu.F0, unix.F0) and 120 < linx.F0 < 150
    assert 0.84 < linx.SCALE < 0.9 and linx.VOWEL_MIX >= 0.5
    names = " ".join(linx.TONES + linx.HISS + linx.CLICKS + [linx.DRUM]).lower()
    assert not any(k in names for k in ("android", "quest", "steam"))
