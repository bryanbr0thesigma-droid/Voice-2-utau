import numpy as np

from voice2utau import audio, sfxbank


def _tone(path, hz=440.0, dur=0.5, lead=0.0, amp=0.5):
    t = np.arange(int(audio.SR_BANK * dur)) / audio.SR_BANK
    x = np.concatenate([np.zeros(int(audio.SR_BANK * lead)), (amp * np.sin(2 * np.pi * hz * t))]).astype("float32")
    audio.write_wav(path, x, audio.SR_BANK)


def test_names():
    assert sfxbank.clean_name("Windows XP Ding", "winxp") == "ding"
    assert sfxbank.clean_name("Windows Logon", "win7") == "logon"
    assert sfxbank._safe("Critical Stop!") == "critical_stop"


def test_build_dedupes_shares_files_and_marks_one_shot(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _tone(a / "Ding.wav", lead=0.3)
    _tone(b / "DING.WAV", lead=0.3)                       # same audio in another group
    _tone(b / "Windows Chord.wav", hz=220.0)
    audio.write_wav(b / "quiet.wav", np.zeros(4410, dtype="float32"), audio.SR_BANK)
    (b / "broken.wav").write_bytes(b"nope")
    r = sfxbank.build([("win31", a), ("win95", b)], tmp_path / "out", "T")
    root = tmp_path / "out" / "T"
    assert r["sounds"] == 2 and r["aliases"] == 3 and len(r["skipped"]["silent"]) == 1 and len(r["skipped"]["unreadable"]) == 1
    lines = (root / "oto.ini").read_bytes().decode("cp932").split("\r\n")
    by = {l.split("=", 1)[1].split(",")[0]: l for l in lines if l}
    assert set(by) == {"win31_ding", "win95_ding", "win95_chord"}
    assert by["win31_ding"].split("=")[0] == by["win95_ding"].split("=")[0]            # one file, two aliases
    f = by["win31_ding"].split(",")
    dur = audio.duration(root / "win31_ding.wav") * 1000
    assert abs(float(f[3]) + dur) < 0.2 and float(f[2]) > dur - 15                    # whole clip is the fixed 'consonant'
    assert dur < 560                                                                  # 300 ms of leading silence trimmed
    frq = (root / "win31_ding_wav.frq").read_bytes()
    assert frq[:8] == b"FREQ0003" and len(frq) == 40 + 16 * (len(frq) - 40) // 16
    peak = float(np.abs(audio.read_wav(root / "win31_ding.wav")[0]).max())
    assert abs(20 * np.log10(peak) + 3.0) < 0.3
