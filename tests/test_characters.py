import numpy as np

from voice2utau import characters, synthbank
from voice2utau.morae import MORAE


def test_pulse_table_has_the_duty_cycle_gaps():
    m = np.abs(np.fft.rfft(characters.pulse_table(0.25)))
    assert m[4] < 1e-3 * m[1] and m[8] < 1e-3 * m[1] and m[3] > 0.1 * m[1]


def test_lfsr_noise_is_one_bit_and_balanced():
    x = characters.lfsr_noise()(20000, np.random.default_rng(0))
    assert set(np.unique(x)) <= {-1.0, 1.0} and abs(x.mean()) < 0.1


def test_bako_whoosh_arrives_before_the_syllable(tmp_path):
    post = characters.bako_post(lambda n, r: r.standard_normal(n))
    y = np.zeros(int(0.5 * synthbank.SR)); c = int(0.1 * synthbank.SR); y[c:] = np.sin(np.arange(len(y) - c) / 10)
    y2, c2 = post(y, c, MORAE["ka"], np.random.default_rng(0))
    assert c2 > c and len(y2) > len(y) and np.abs(y2[:c2 - c]).max() > 0              # material added in front


def test_hooks_are_restored_after_building(tmp_path):
    characters.build_ninchan(tmp_path, clicks=[])
    assert synthbank.SOURCE_FN is None and synthbank.NOISE_FN is None and synthbank.POST_FN is None


def test_airy_source_mixes_soft_tone_and_breath():
    from voice2utau import ninchan
    from voice2utau.characters import F0, SR, airy_source
    t = np.arange(ninchan.N) / ninchan.N
    table = np.sign(np.sin(2 * np.pi * t))                    # bright square: the tilt must round it off
    rng = np.random.default_rng(0)
    tone = airy_source(table, breath=0.0)(SR, rng)
    X = np.abs(np.fft.rfft(tone)); f = np.fft.rfftfreq(SR, 1 / SR)
    h = lambda k: X[np.argmin(np.abs(f - k * F0))]
    assert h(9) / h(1) < (1 / 9) * 0.5                        # far softer than a square's 1/9
    whisper = airy_source(table, breath=1.0)(SR, rng)
    W = np.abs(np.fft.rfft(whisper))
    assert W[f > 2000].sum() > W[(f > 50) & (f < 600)].sum()  # breath lives up high
    assert np.abs(airy_source(table, 0.35)(1000, rng)).max() <= 1.0 + 1e-9
