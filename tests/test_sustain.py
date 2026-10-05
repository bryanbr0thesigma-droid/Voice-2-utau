import json

import numpy as np
import pytest

pytest.importorskip("parselmouth")
pytest.importorskip("torchaudio")

from voice2utau import audio, sustain
from voice2utau.crossling import oto_line, read_oto
from voice2utau.phonemes import Phone

SR = 44100


def vowel(seconds, f0=280.0, formants=((700, 80), (1200, 90))):
    """Harmonic pulse train through two resonators: a steady synthetic vowel."""
    from scipy.signal import lfilter
    n = int(seconds * SR)
    x = np.zeros(n)
    x[::int(SR / f0)] = 1.0
    for f, bw in formants:
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * f / SR
        x = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def test_find_held_finds_steady_vowel():
    x = np.concatenate([np.zeros(SR // 5, np.float32), vowel(0.6), np.zeros(SR // 5, np.float32)])
    held = sustain.find_held(x, SR, [Phone("ɑ", 0.3, 0.5, 0.9)])
    assert len(held) == 1 and held[0]["vowel"] == "aa"
    assert held[0]["end"] - held[0]["start"] >= sustain.MIN_HELD_S


def test_find_held_ignores_short_vowel():
    x = np.concatenate([np.zeros(SR // 5, np.float32), vowel(0.15), np.zeros(SR // 5, np.float32)])
    assert sustain.find_held(x, SR, [Phone("a", 0.22, 0.32, 0.9)]) == []


def test_extend_inserts_body_and_keeps_pitch():
    clip = vowel(0.2)
    body = sustain.Body("aa", vowel(0.4), "rvc")
    r = sustain.extend(clip, SR, 50.0, "aa", {"aa": [body]}, scale=20.0, bank_f0=280.0)
    assert r is not None
    assert 300 < r.inserted_ms < 420
    f0 = audio.median_f0(r.audio, SR)
    assert abs(f0 - 280) < 8


def test_extend_returns_none_without_body_or_room():
    assert sustain.extend(vowel(0.2), SR, 50.0, "aa", {}, 20.0, 280.0) is None
    body = sustain.Body("aa", vowel(0.4), "rvc")
    assert sustain.extend(vowel(0.2), SR, 190.0, "aa", {"aa": [body]}, 20.0, 280.0) is None


def test_apply_updates_all_aliases_and_oto(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    audio.write_wav(bank / "cv_k_aa.wav", vowel(0.2), SR)
    audio.write_wav(bank / "cv_s_ow.wav", vowel(0.2), SR)
    lines = [oto_line("cv_k_aa", "k aa", [0, 100, -200, 50, 16]), oto_line("cv_k_aa", "ka", [0, 100, -200, 50, 16]),
             oto_line("cv_s_ow", "s ow", [0, 100, -200, 50, 16])]
    (bank / "oto.ini").write_bytes(("\n".join(lines) + "\n").encode("cp932"))
    (bank / "readme.txt").write_text("x", encoding="utf-8")
    rep = {"bank_f0_hz": 280.0, "samples": [{"key": "cv_k_aa", "source": "recorded"}, {"key": "cv_s_ow", "source": "recorded"}]}
    (bank / "voice2utau_report.json").write_text(json.dumps(rep), encoding="utf-8")
    s = sustain.apply(bank, tmp_path / "out", [sustain.Body("aa", vowel(0.4), "rvc"), sustain.Body("ao", vowel(0.4), "rvc")])
    assert s["extended"] == 2
    oto = (tmp_path / "out" / "oto.ini").read_bytes().decode("cp932").splitlines()
    assert len(oto) == 3
    cut = [float(l.split(",")[3]) for l in oto]
    assert all(c < -200 for c in cut)                       # longer than the original 200 ms
    n = len(audio.read_wav(tmp_path / "out" / "cv_k_aa.wav")[0]) / SR * 1000
    assert abs(-cut[0] - n) < 3                             # cutoff = new clip length
    assert audio.read_wav(bank / "cv_k_aa.wav")[0].shape[0] == int(0.2 * SR)    # input bank untouched
    assert read_oto(tmp_path / "out" / "oto.ini")["cv_s_ow"][0] == "s ow"
