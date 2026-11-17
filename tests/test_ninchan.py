import numpy as np

from voice2utau import audio, ninchan, synthbank
from voice2utau.morae import MORAE


def _tone(path, f0, dur=0.5, harmonics=12):
    t = np.arange(int(audio.SR_BANK * dur)) / audio.SR_BANK
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, harmonics + 1))
    audio.write_wav(path, (0.4 * x / np.abs(x).max()).astype("float32"), audio.SR_BANK)


def test_wavetable_is_one_clean_period():
    sr, f0 = audio.SR_BANK, 440.0
    t = np.arange(int(sr * 0.3)) / sr
    x = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
    wt = ninchan.wavetable(x.astype("float32"), sr, 1000, f0)
    m = np.abs(np.fft.rfft(wt))
    assert m[1] > 5 * m[3:].max() and m[2] > 5 * m[3:].max()                 # energy only on harmonics 1 and 2
    assert ninchan.richness(wt) == 2


def test_flatten_gives_a_voice_like_tilt():
    wt = np.random.default_rng(0).standard_normal(ninchan.N)
    y = ninchan.flatten(wt, keep=0.0)
    m = np.abs(np.fft.rfft(y))
    assert m[1:4].mean() > 10 * m[30:40].mean()                               # low harmonics dominate, like a voice


def test_build_uses_nintendo_sources_and_restores_the_synth(tmp_path):
    d = tmp_path / "snd"
    d.mkdir()
    _tone(d / "chime.wav", 523.25)
    rng = np.random.default_rng(1)
    audio.write_wav(d / "click.wav", (rng.standard_normal(4410) * 0.3).astype("float32"), audio.SR_BANK)
    with ninchan.nintendo_voice([d], top=1) as info:
        assert synthbank.SOURCE_FN is not None and info["carriers"][0][0] == "chime.wav"
        y, pre = synthbank.synth(MORAE["ka"], np.random.default_rng(2))
        assert np.isfinite(y).all() and np.abs(y).max() > 0.1
    assert synthbank.SOURCE_FN is None and synthbank.NOISE_FN is None
