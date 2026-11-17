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


def test_kaso_post_lands_a_pop_on_the_vowel_onset():
    from voice2utau.characters import SR, kaso_post
    y = np.zeros(int(0.5 * SR)); c = int(0.1 * SR)
    y[c:] = 0.5 * np.sin(2 * np.pi * 220 * np.arange(len(y) - c) / SR)    # low tone: inside the ducked band
    pop = np.ones(2000)
    out, c2 = kaso_post(pop, pop_db=-6.0, duck_db=-120.0, bloom_db=0.0)(y, c, None, np.random.default_rng(0))
    assert c2 == c
    a = c - int(0.004 * SR)
    assert np.allclose(out[:a], 0)
    under = out[c + 10:c + 400]                               # vowel ducked away: only the pop is left here
    assert abs(np.sqrt((under ** 2).mean()) / (0.5 / np.sqrt(2)) - 10 ** (-6 / 20)) < 0.02
    assert np.allclose(out[c + int(0.06 * SR):], y[c + int(0.06 * SR):])   # vowel back in full by 60 ms
