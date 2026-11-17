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
    assert e(0.6, 0.9) > e(0.03, 0.2) and e(0.6, 0.9) > e(1.6, 1.95)  # swells, then fades
    X = np.abs(np.fft.rfft(ir)); f = np.fft.rfftfreq(len(ir), 1 / SR)
    band = (f > 300) & (f < 700)
    assert X[band].max() < 8 * np.median(X[band])             # no sharp G/D peaks left to clash with the song


def test_dream_keeps_dry_and_adds_tail():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(SR) * 0.1).astype("float32")
    ir = quest.startup_ir(_startup(), SR)
    y = quest.dream(x, SR, ir, wet_db=-6.0, contained=False)
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


def test_contain_fades_the_tail_after_she_stops():
    dry = np.zeros(SR * 3); dry[:SR] = 0.3
    wet = np.ones(SR * 3)
    g = quest.contain(wet, dry, SR, hold=0.35, release=0.4)
    assert g[int(0.5 * SR)] > 0.99                            # while singing
    assert g[int(1.3 * SR)] > 0.95                            # held just after
    assert g[int(1.9 * SR)] < 0.01                            # gone 0.75 s after she stops
    assert np.all(np.diff(g[int(1.3 * SR):int(1.9 * SR)]) <= 1e-9)


def test_dream_keeps_reverb_out_of_the_low_mids():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(SR) * 0.1).astype("float32")
    ir = quest.startup_ir(_startup(), SR)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    def lowmid_share(**kw):
        wet = quest.dream(x, SR, ir, contained=False, **kw)[:len(x)] - x
        W = np.abs(np.fft.rfft(wet)) ** 2
        return W[(f > 150) & (f < 300)].sum() / W.sum()
    assert lowmid_share(wet_hp=550.0) < 0.05 * lowmid_share(wet_hp=0)     # at least 13 dB less body


def test_startup_ir_has_air_not_a_midrange_fog():
    ir = quest.startup_ir(_startup(), SR)
    X = np.abs(np.fft.rfft(ir)) ** 2; f = np.fft.rfftfreq(len(ir), 1 / SR)
    mid = X[(f > 500) & (f < 1000)].sum() / 500
    air = X[(f > 3000) & (f < 6000)].sum() / 3000
    assert air > 0.7 * mid                                    # per Hz, about as much air as midrange
