import numpy as np

from voice2utau import andro, puresu, unix


def test_mulaw_grit_keeps_the_signal_and_has_256_levels():
    t = np.arange(4410) / 44100
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    out = unix.mulaw(8)(y)
    assert np.abs(out - y).max() < 0.02                       # close to the signal...
    assert 0 < np.abs(out - y).mean()                         # ...but quantised
    assert len(np.unique(np.round(out, 9))) <= 256
    quiet = unix.mulaw(8)(0.01 * np.sin(2 * np.pi * 220 * t))
    assert np.abs(quiet).max() > 0.008                        # mu-law keeps quiet detail (unlike linear 8-bit)


def test_unix_is_male_and_distinct_from_the_other_men():
    assert unix.TONES and unix.HISS and unix.CLICKS and unix.DIRECT
    assert unix.F0 < 261.63 and unix.SCALE < 1.0
    assert unix.F0 not in (andro.MALE_F0, puresu.F0)
    assert not any(n.startswith("linux") for n in unix.TONES + unix.HISS + unix.CLICKS)
