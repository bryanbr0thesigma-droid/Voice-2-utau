import numpy as np

from voice2utau import quest
from voice2utau.sing import Note

SR = 44100


def _note(lyric, start, dur, num=60):
    return Note(lyric=lyric, length=480, notenum=num, tempo=120.0, start=start, dur=dur)


def _voice(sec=2.0):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")


def test_passthrough_keeps_level_and_steps_at_frame_rate():
    x = _voice()
    y = quest.passthrough(x, SR, fps=36.0, grain=0.0, lag_db=-120.0, top_hz=20000.0)
    assert abs(np.sqrt((y ** 2).mean()) / np.sqrt((x ** 2).mean()) - 1) < 1e-3
    assert not np.allclose(y, x, atol=1e-2)


def test_phrases_split_on_rests():
    notes = [_note("a", 0.0, 0.5), _note("i", 0.5, 0.5), _note("R", 1.0, 0.5), _note("u", 1.5, 0.5)]
    assert quest.phrases(notes) == [(0.0, 1.0), (1.5, 2.0)]


def test_emerge_starts_in_passthrough_and_resolves_clean():
    clean = np.ones(SR * 2, dtype="float32"); pt = np.zeros_like(clean)
    y = quest.emerge(clean, pt, SR, [_note("a", 0.5, 1.0)], ramp=0.4, lead=0.1)
    assert y[int(0.45 * SR)] < 0.01 and y[int(0.95 * SR)] > 0.99 and y[int(0.2 * SR)] > 0.99


def test_guardian_only_flares_on_top_notes():
    x = _voice(2.0)
    notes = [_note("a", 0.0, 0.9, 60), _note("i", 1.0, 0.9, 67)]
    assert [n.notenum for n in quest.high_notes(notes)] == [67]
    sh = np.random.default_rng(0).standard_normal(SR).astype("float32")
    y = quest.guardian(x, SR, notes, sh)
    assert np.allclose(y[:int(0.9 * SR)], x[:int(0.9 * SR)], atol=1e-6)
    assert not np.allclose(y[int(1.2 * SR):int(1.8 * SR)], x[int(1.2 * SR):int(1.8 * SR)], atol=1e-3)
