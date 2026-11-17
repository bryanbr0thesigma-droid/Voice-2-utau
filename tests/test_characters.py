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
