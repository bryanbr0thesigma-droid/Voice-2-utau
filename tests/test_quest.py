import numpy as np
import pytest

from voice2utau import quest

SR = 22050


def _startup(sec=3.0):
    t = np.arange(int(sec * SR)) / SR
    env = np.minimum(1, t / 0.7) * np.exp(-np.maximum(0, t - 0.7) / 0.6)
    return (env * (np.sin(2 * np.pi * 392 * t) + np.sin(2 * np.pi * 588 * t))).astype("float32")


def test_startup_ir_keeps_the_swell_but_not_the_chord():
    ir = quest.startup_ir(_startup(), SR)
    assert abs(np.sqrt((ir.astype(float) ** 2).sum()) - 1) < 1e-4
    assert np.allclose(ir[:int(0.02 * SR)], 0)                # pre-delay
    e = lambda a, b: np.sqrt((ir[int(a * SR):int(b * SR)] ** 2).mean())
    assert e(0.6, 0.9) > e(0.03, 0.2) and e(0.6, 0.9) > e(2.4, 2.8)   # swells, then fades
    X = np.abs(np.fft.rfft(ir)); f = np.fft.rfftfreq(len(ir), 1 / SR)
    band = (f > 300) & (f < 700)
    assert X[band].max() < 8 * np.median(X[band])             # no sharp G/D peaks left to clash with the song


def test_dream_keeps_dry_and_adds_tail():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(SR) * 0.1).astype("float32")
    ir = quest.startup_ir(_startup(), SR)
    y = quest.dream(x, SR, ir, wet_db=-6.0)
    assert len(y) == len(x) + len(ir) - 1
    assert np.sqrt((y[len(x):] ** 2).mean()) > 0            # reverb tail rings past the end
    wet = y[:len(x)] - x
    assert abs(20 * np.log10(np.sqrt((wet ** 2).mean()) / np.sqrt((x ** 2).mean())) + 6) < 0.1


def test_halo_is_an_octave_up():
    pytest.importorskip("pyworld")
    t = np.arange(SR) / SR
    x = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 12)).astype("float32") * 0.2
    y = quest.halo(x, SR)
    X = np.abs(np.fft.rfft(y[SR // 4:])); f = np.fft.rfftfreq(len(y) - SR // 4, 1 / SR)
    h = lambda hz: X[np.abs(f - hz) < 15].max()
    assert h(1760) > 20 * h(1540)                             # 1760 = 4 x 440; 1540 = 7 x 220 only
