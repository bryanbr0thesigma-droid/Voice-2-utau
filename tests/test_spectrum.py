import numpy as np

from voice2utau import audio, spectrum


def _tone_set(d, rolloff, n=3, seed=0):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(int(audio.SR_BANK * 0.8)) / audio.SR_BANK
    for i in range(n):
        x = sum(np.sin(2 * np.pi * 170 * k * t + rng.uniform(0, 6)) * k ** -rolloff for k in range(1, 40))
        audio.write_wav(d / f"{i}.wav", (0.2 * x / np.abs(x).max()).astype("float32"), audio.SR_BANK)
    return sorted(d.glob("*.wav"))


def test_eq_moves_a_dark_set_toward_the_reference(tmp_path):
    ref = _tone_set(tmp_path / "ref", 0.5)                 # bright
    dark = _tone_set(tmp_path / "dark", 2.0, seed=1)       # dull
    (tmp_path / "dark" / "oto.ini").write_text("x")
    g = spectrum.match_dir(tmp_path / "dark", tmp_path / "out", ref)
    assert g[3] > 3                                         # boosts 2 kHz
    assert (tmp_path / "out" / "oto.ini").exists()          # non-wav files are carried over
    gap = lambda files: np.abs(spectrum.smooth_db(spectrum.ltas(ref)) - spectrum.smooth_db(spectrum.ltas(files)))[50:400].mean()
    assert gap(sorted((tmp_path / "out").glob("*.wav"))) < gap(dark)
    x, _ = audio.read_wav(tmp_path / "out" / "0.wav")
    x0, _ = audio.read_wav(tmp_path / "dark" / "0.wav")
    assert abs(np.sqrt((x ** 2).mean()) / np.sqrt((x0 ** 2).mean()) - 1) < 0.05 and len(x) == len(x0)


def test_toward_is_between_reference_and_other(tmp_path):
    ref = _tone_set(tmp_path / "ref", 1.0)
    bright = _tone_set(tmp_path / "bright", 0.6, seed=2)
    src = _tone_set(tmp_path / "src", 1.4, seed=3)
    _, g_ref = spectrum.gain_db(ref, src)
    _, g_mid = spectrum.gain_db(ref, src, toward=bright)
    assert g_mid[90:230].mean() > g_ref[90:230].mean() + 0.5


def test_consonant_part_gets_a_capped_gain(tmp_path):
    src = tmp_path / "src"
    ref = _tone_set(tmp_path / "ref", 0.5)
    _tone_set(src, 2.0, n=1)                                         # dull; the EQ wants a big boost
    (src / "oto.ini").write_bytes(b"0.wav=ma,0,100,-400,300,50\r\n")   # preutterance 300 ms
    f, g = spectrum.gain_db(ref, sorted(src.glob("*.wav")))
    assert spectrum.read_pre_ms(src / "oto.ini") == {"0.wav": 300.0}
    spectrum.apply_eq(src, tmp_path / "capped", f, g, cons_cap_db=0.0)
    spectrum.apply_eq(src, tmp_path / "plain", f, g, cons_cap_db=None)
    a, _ = audio.read_wav(tmp_path / "capped" / "0.wav")
    b, _ = audio.read_wav(tmp_path / "plain" / "0.wav")
    hf = lambda y: np.abs(np.fft.rfft(y[: int(0.2 * audio.SR_BANK)]))[200:1500].sum()
    assert hf(a) < 0.7 * hf(b)                                       # before the preutterance: no presence boost
