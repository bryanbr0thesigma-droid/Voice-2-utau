import numpy as np
import parselmouth
from scipy.signal import stft

from voice2utau import audio, synthbank
from voice2utau.morae import MORAE


def _centroid(x, a, b):
    f, _, Z = stft(x[a:b], synthbank.SR, nperseg=512, boundary=None)
    P = (np.abs(Z) ** 2).mean(1)
    return float((f * P).sum() / P.sum())


def test_every_unit_is_synthesised_at_exact_pitch_and_is_clean():
    rng = np.random.default_rng(3)
    for key in ("a", "ka", "shi", "tsu", "nn", "ryo", "fu", "nya", "wa"):
        y, split = synthbank.synth(MORAE[key], rng)
        assert np.isfinite(y).all() and 0.1 < np.abs(y).max() <= 0.81 and len(y) / synthbank.SR > 0.3, key
        assert 0.01 <= split <= 0.13, key
    y, _ = synthbank.synth(MORAE["a"], rng)
    p = parselmouth.Sound(y.astype("float64"), synthbank.SR).to_pitch(0.01, 100, 600)
    f0 = np.nanmedian([p.get_value_at_time(t) for t in np.arange(0.15, 0.4, 0.02)])
    assert abs(f0 - synthbank.F0) < 1.5


def test_fricatives_have_the_right_noise_band_and_vowels_differ():
    rng = np.random.default_rng(3)
    s, ps = synthbank.synth(MORAE["sa"], rng)
    sh, psh = synthbank.synth(MORAE["sha"], rng)
    assert _centroid(s, 200, int(ps * synthbank.SR) - 200) > 5500
    assert 2500 < _centroid(sh, 200, int(psh * synthbank.SR) - 200) < 5500
    r = {}
    for v in "iuea":                                                  # share of energy in the F2/F3 region vs F1/F2 region
        y, _ = synthbank.synth(MORAE[v], rng)
        f, _, Z = stft(y[int(0.15 * synthbank.SR):int(0.3 * synthbank.SR)], synthbank.SR, nperseg=2048, boundary=None)
        P = (np.abs(Z) ** 2).mean(1)
        r[v] = 10 * np.log10(P[(f >= 1800) & (f < 3300)].sum() / P[(f >= 600) & (f < 1600)].sum())
    assert r["i"] > r["e"] > r["u"] > r["a"] and r["i"] - r["a"] > 20          # high F2 for i/e, low F2 for u/a


def test_build_writes_the_whole_cv_set(tmp_path):
    root = synthbank.build(tmp_path, "T")
    lines = [l for l in (root / "oto.ini").read_bytes().decode("cp932").splitlines() if "=" in l]
    assert len(lines) == len(MORAE) == 101
    assert {l.split("=")[1].split(",")[0] for l in lines} >= {"あ", "か", "しゃ", "つ", "ん", "りょ"}
    assert all((root / l.split("=")[0]).exists() and (root / (l.split("=")[0][:-4] + "_wav.frq")).exists() for l in lines)
    ka = next(l for l in lines if l.startswith("ka.wav")).split(",")
    assert 55 <= float(ka[4]) <= 70                                           # preutterance ~ the consonant length
