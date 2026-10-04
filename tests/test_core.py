import zipfile
from pathlib import Path

import numpy as np
import pytest

from voice2utau import audio, bank, rvc
from voice2utau.extract import Clip, build_candidates
from voice2utau.ingest import IngestError, safe_extract_zip
from voice2utau.morae import MORAE, classify, lookup
from voice2utau.phonemes import Phone
from voice2utau.templates import alias_to_mora, pick_espeak_voice

SR = audio.SR_BANK


def tone(dur=0.3, f=200.0, amp=0.3):
    t = np.arange(int(SR * dur)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


# ---- morae -----------------------------------------------------------------
def test_inventory_unique_and_ascii_keys():
    assert len(MORAE) == 101
    assert len({m.kana for m in MORAE.values()}) == len(MORAE)
    assert all(k.isascii() for k in MORAE)


@pytest.mark.parametrize("cls,v,pal,key", [
    ("k", "a", False, "ka"), ("s", "i", False, "shi"), ("k", "a", True, "kya"),
    ("t", "i", False, "chi"), ("h", "u", False, "fu"), ("sh", "a", False, "sha")])
def test_lookup(cls, v, pal, key):
    assert lookup(cls, v, pal).key == key


def test_ipa_classification():
    assert classify("tʃ").value == "ch" and classify("iː").value == "i"
    assert classify("oʊ").value == "o" and classify("ŋ").kind == "N"
    assert classify("nʲ").palatal and classify("ə").kind == "X"


def test_alias_mapping_accepts_kana_katakana_romaji():
    assert alias_to_mora("か").key == "ka" and alias_to_mora("カ").key == "ka"
    assert alias_to_mora("- き").key == "ki" and alias_to_mora("shi").key == "shi"
    assert alias_to_mora("a ka") is None


# ---- candidates ------------------------------------------------------------
def test_build_candidates_cv_and_palatal():
    ph = [Phone("k", 1.00, 1.04, .9), Phone("a", 1.10, 1.14, .9),
          Phone("k", 2.00, 2.04, .9), Phone("j", 2.06, 2.08, .9), Phone("u", 2.14, 2.20, .9),
          Phone("a", 5.0, 5.1, .9)]
    keys = [c.mora.key for c in build_candidates(ph, "s")]
    assert keys == ["ka", "kyu", "a"]


def test_build_candidates_ignores_cluster_without_vowel():
    ph = [Phone("s", 1.0, 1.05, .9), Phone("t", 1.06, 1.1, .9)]
    assert build_candidates(ph, "s") == []


# ---- oto.ini / bank --------------------------------------------------------
def test_oto_line_and_sjis_roundtrip(tmp_path):
    clip = Clip(MORAE["ka"], tone(0.3), 0.08, 200.0)
    line = bank.oto_line(clip)
    assert line.startswith("ka.wav=か,0,") and line.split(",")[3].startswith("-300")
    root = bank.write_bank({"ka": clip}, tmp_path, "テスト Voice", {"x": 1})
    text = (root / "oto.ini").read_bytes().decode("cp932")
    assert text.strip() == line
    assert "テスト" in (root / "character.txt").read_bytes().decode("cp932")
    assert (root / "ka.wav").exists()
    z = bank.zip_bank(root, tmp_path / "b.zip")
    assert any(n.endswith("oto.ini") for n in zipfile.ZipFile(z).namelist())


def test_safe_name_strips_path_chars():
    assert "/" not in bank.safe_name("../../etc/passwd") and bank.safe_name("   ") == "voice"


# ---- zip safety ------------------------------------------------------------
def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for n, d in entries:
            z.writestr(n, d)


def test_zip_slip_rejected(tmp_path):
    make_zip(tmp_path / "e.zip", [("../evil.wav", b"x")])
    with pytest.raises(IngestError):
        safe_extract_zip(tmp_path / "e.zip", tmp_path / "out", {".wav"})
    assert not (tmp_path / "evil.wav").exists()


def test_zip_filters_extensions_and_junk(tmp_path):
    make_zip(tmp_path / "z.zip", [("a/ok.wav", b"1"), ("a/run.sh", b"2"), ("__MACOSX/x.wav", b"3"), ("a/._ok.wav", b"4")])
    out = safe_extract_zip(tmp_path / "z.zip", tmp_path / "out", {".wav"})
    assert [p.name for p in out] == ["ok.wav"]


def test_not_a_zip(tmp_path):
    (tmp_path / "n.zip").write_bytes(b"nope")
    with pytest.raises(IngestError):
        safe_extract_zip(tmp_path / "n.zip", tmp_path / "o")


# ---- RVC batching ----------------------------------------------------------
class FakeBackend(rvc.RVCBackend):
    """Returns the input at 40 kHz, like many real RVC models."""

    def __init__(self):
        self.calls = []

    def convert(self, src, dst, transpose):
        from scipy.signal import resample_poly
        x, sr = audio.read_wav(src)
        audio.write_wav(dst, resample_poly(x, 40000, sr).astype(np.float32) * 0.5, 40000)
        self.calls.append(transpose)


def test_convert_clips_batches_and_splits_correctly():
    freqs = [120 + 20 * i for i in range(20)]            # distinct pitch identifies each clip
    keys = list(MORAE)[:20]
    clips = [Clip(MORAE[k], tone(0.2 + 0.01 * i, f), 0.05, f) for i, (k, f) in enumerate(zip(keys, freqs))]
    be = FakeBackend()
    out, warns = rvc.convert_clips(clips, be, transpose=3)
    assert len(out) == 20 and len(be.calls) == 2 and set(be.calls) == {3}   # batches of 16 + 4
    assert not warns
    for c, f in zip(out, freqs):
        assert c.source == "rvc" and abs(c.f0 - f) < 8, (c.mora.key, c.f0, f)


def test_command_backend_validates_and_runs(tmp_path):
    with pytest.raises(rvc.RVCError):
        rvc.CommandBackend("echo hi", tmp_path / "m.pth", None)
    (tmp_path / "m.pth").write_bytes(b"x")
    fake = Path(__file__).parent / "fake_rvc.py"
    b = rvc.CommandBackend(f"python {fake} {{input}} {{output}} {{transpose}}", tmp_path / "m.pth", None)
    src, dst = tmp_path / "i.wav", tmp_path / "o.wav"
    audio.write_wav(src, tone(), SR)
    b.convert(src, dst, 2)
    assert audio.read_wav(dst)[1] == 40000
    bad = rvc.CommandBackend("false {input} {output}", tmp_path / "m.pth", None)
    with pytest.raises(rvc.RVCError):
        bad.convert(src, tmp_path / "x.wav", 0)


def test_make_backend_requires_model():
    with pytest.raises(rvc.RVCError):
        rvc.make_backend(None, None)


# ---- audio helpers ---------------------------------------------------------
def test_flatten_pitch_reaches_target():
    x = tone(0.6, 180)
    y, changed = audio.flatten_pitch(x, SR, 200.0)
    assert changed and abs(audio.median_f0(y, SR) - 200) < 3
    _, changed = audio.flatten_pitch(x, SR, 600.0)      # > 7 semitones away: refuse
    assert not changed


def test_chunking_cuts_in_quiet_gaps():
    sr = 16000
    x = np.concatenate([np.ones(sr * 19) * 0.3, np.zeros(sr // 2), np.ones(sr * 30) * 0.3]).astype(np.float32)
    b = audio.chunk_boundaries(x, sr, target_s=20)
    assert len(b) >= 2 and 19 * sr <= b[0][1] <= int(19.6 * sr)
    assert b[0][0] == 0 and b[-1][1] == len(x) and all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))


def test_pick_espeak_voice_tracks_target_pitch():
    assert pick_espeak_voice(100)[0] == "ja" and pick_espeak_voice(230)[0] == "ja+f3"


def test_language_tags_from_zip_paths():
    from voice2utau.ingest import detect_lang
    assert detect_lang("de/line1.wav") == "de" and detect_lang("lines/cyn_de_004.wav") == "de"
    assert detect_lang("English/a.mp3") == "en" and detect_lang("Cyn (German) 01.wav") == "de"
    assert detect_lang("deep/a.wav") is None and detect_lang("lines/design.wav") is None


def test_zip_sources_get_language_from_folders(tmp_path):
    from voice2utau.ingest import prepare_sources
    wav = tmp_path / "t.wav"
    audio.write_wav(wav, tone(0.5), SR)
    with zipfile.ZipFile(tmp_path / "v.zip", "w") as z:
        z.write(wav, "en/a.wav"); z.write(wav, "de/b.wav"); z.write(wav, "c.wav")
    got = {s.original: s.lang for s in prepare_sources(tmp_path / "v.zip", tmp_path / "w")}
    assert got == {"a.wav": "en", "b.wav": "de", "c.wav": "en"}
    got = {s.original: s.lang for s in prepare_sources(tmp_path / "v.zip", tmp_path / "w2", default_lang="de")}
    assert got == {"a.wav": "en", "b.wav": "de", "c.wav": "de"}
