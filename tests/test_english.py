import numpy as np
import pytest

from voice2utau import audio, bank, english, pipeline, profiles
from voice2utau.extract import Clip
from voice2utau.phonemes import Phone
from voice2utau.templates import EspeakTemplate, alias_to_mora, estimate_split, pick_espeak_voice

SR = audio.SR_BANK


def tone(dur=0.3, f=200.0, amp=0.3):
    t = np.arange(int(SR * dur)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_inventory_shape():
    u = english.UNITS
    kinds = {k: sum(m.kind == k for m in u.values()) for k in ("CV", "VC", "V")}
    assert kinds == {"CV": 23 * 15, "VC": 21 * 15, "V": 15} and len(u) == 675
    assert "cv_ng_aa" not in u and "vc_aa_hh" not in u          # impossible English combinations
    assert u["cv_k_ae"].kana == "k ae" and u["vc_ae_t"].kana == "ae t" and u["v_ae"].kana == "- ae"
    assert all(k.isascii() and "/" not in k for k in u)


@pytest.mark.parametrize("ipa,kind,val", [
    ("tʃ", "C", "ch"), ("dʒ", "C", "jh"), ("θ", "C", "th"), ("ŋ", "C", "ng"), ("ɹ", "C", "r"), ("j", "C", "y"),
    ("iː", "V", "iy"), ("ɑː", "V", "aa"), ("æ", "V", "ae"), ("ə", "V", "ah"), ("ɚ", "V", "er"),
    ("aɪ", "V", "ay"), ("oʊ", "V", "ow"), ("ɔɪ", "V", "oy"), ("ɑːɹ", "V", "aa"), ("ʔ", "X", ""), ("?", "X", "")])
def test_classify(ipa, kind, val):
    s = english.classify(ipa)
    assert (s.kind, s.value) == (kind, val)


def test_build_candidates_cv_vc_and_initial_vowel():
    # "cat" = k ae t, then a pause, then an utterance-initial vowel "ih"
    ph = [Phone("k", 1.00, 1.05, .9), Phone("æ", 1.08, 1.16, .9), Phone("t", 1.20, 1.25, .9),
          Phone("ɪ", 3.0, 3.1, .8)]
    got = {c.mora.key: c for c in english.build_candidates(ph, "s")}
    assert set(got) == {"cv_k_ae", "vc_ae_t", "v_ih"}
    cv, vc = got["cv_k_ae"], got["vc_ae_t"]
    assert cv.start < 1.0 and cv.start < cv.split < vc.end                     # CV starts at the consonant
    assert vc.split == pytest.approx((1.16 + 1.20) / 2) and vc.end > 1.25      # VC ends after the consonant
    assert cv.conf == pytest.approx(.9)


def test_non_adjacent_phones_do_not_pair():
    ph = [Phone("k", 1.0, 1.05, .9), Phone("æ", 1.6, 1.7, .9)]               # 0.55 s apart
    assert [c.mora.key for c in english.build_candidates(ph, "s")] == ["v_ae"]


def test_vc_oto_line_semantics():
    clip = Clip(english.UNITS["vc_ae_t"], np.zeros(int(SR * 0.3), np.float32), 0.18, 200.0)
    f = bank.oto_line(clip).split("=")[1].split(",")
    assert f[0] == "ae t" and f[1] == "0" and float(f[3]) == pytest.approx(-300, abs=1)
    assert float(f[4]) == pytest.approx(180, abs=1)                          # preutterance = vowel length
    assert float(f[2]) >= float(f[4])


def test_english_alias_parsing_for_custom_banks():
    u = english.UNITS
    assert alias_to_mora("K  AE", u).key == "cv_k_ae"
    assert alias_to_mora("ae t", u).key == "vc_ae_t"
    assert alias_to_mora("- ae", u).key == "v_ae" and alias_to_mora("ka", u) is None


def test_default_split_and_espeak_strings():
    assert english.espeak_phonemes(english.UNITS["cv_k_ae"]) == "k'a"
    assert english.espeak_phonemes(english.UNITS["vc_ae_t"]) == "'at"
    assert english.espeak_phonemes(english.UNITS["v_iy"]) == "'i:"
    x = np.zeros(int(SR * 0.3), np.float32)
    assert estimate_split(english.UNITS["vc_ae_t"], x, SR) == pytest.approx(0.2, abs=0.01)
    assert 0.03 < estimate_split(english.UNITS["cv_b_ae"], x, SR) <= 0.15


def test_espeak_voice_table_for_english():
    assert pick_espeak_voice(90, "en")[0] == "en-us" and pick_espeak_voice(230, "en")[0] == "en-us+f3"


def test_every_english_unit_has_espeak_mnemonic():
    for m in english.UNITS.values():
        assert english.espeak_phonemes(m)


@pytest.mark.skipif(__import__("shutil").which("espeak-ng") is None, reason="espeak-ng missing")
def test_espeak_template_produces_every_kind():
    t = EspeakTemplate(180, lang="en")
    for key in ("cv_k_ae", "vc_ae_t", "v_ih", "cv_sh_iy", "vc_ow_n"):
        c = t.get(english.UNITS[key])
        assert c is not None and len(c.audio) > SR * 0.1 and c.f0 and c.source == "template", key
    vc = t.get(english.UNITS["vc_ae_t"])
    assert 0.03 < vc.split_s < len(vc.audio) / SR


def test_profiles():
    assert profiles.get("en").default_validate is False and profiles.get("ja").default_validate is True
    with pytest.raises(ValueError):
        profiles.get("fr")
    with pytest.raises(pipeline.PipelineError):
        pipeline.run(__import__("pathlib").Path("x.mp3"), __import__("pathlib").Path("/tmp/v2u_w"),
                     __import__("pathlib").Path("/tmp/v2u_o"), pipeline.Options(language="fr"))


# ---- German as a fallback source ------------------------------------------------------------------
def test_german_mapping_keeps_close_sounds_and_drops_the_rest():
    ok = {"ɪ": "ih", "iː": "iy", "ʊ": "uh", "ɛ": "eh", "aɪ": "ay", "aʊ": "aw", "ɔʏ": "oy", "ʃ": "sh", "ŋ": "ng"}
    for ipa, arpa in ok.items():
        assert english.classify_de(ipa).value == arpa, ipa
    for ipa in ("ʁ", "x", "ç", "ts", "pf", "y", "ʏ", "ø", "œ", "eː", "oː"):       # no honest English equivalent
        assert english.classify_de(ipa).kind == "X", ipa


def test_german_candidates_are_marked_as_fallback():
    ph = [Phone("ʃ", 1.0, 1.05, .9), Phone("ɪ", 1.08, 1.16, .9), Phone("ʁ", 1.2, 1.25, .9)]
    de = english.build_candidates(ph, "s", "de")
    assert [c.mora.key for c in de] == ["cv_sh_ih"] and de[0].penalty > 0   # ʁ dropped: no vc_ih_r
    assert english.build_candidates(ph, "s", "en")[0].penalty == 0


def test_native_recording_always_beats_german(tmp_path):
    from voice2utau import extract
    key = "cv_k_ae"
    x = np.concatenate([tone(0.3, 200, .3)] * 2)
    wavs = {"en": tmp_path / "en.wav", "de": tmp_path / "de.wav"}
    for p in wavs.values():
        audio.write_wav(p, np.concatenate([np.zeros(SR // 2, np.float32), x, np.zeros(SR // 2, np.float32)]), SR)
    mk = lambda src, conf, pen: extract.Candidate(english.UNITS[key], src, 0.4, 0.8, 0.5, conf, 0.2, 1.0 + conf, pen)
    # German has far higher confidence, yet the English take must win
    got = extract.select_best([mk("en", .72, 0.0), mk("de", .99, 0.1)], wavs, min_conf=.7)
    assert got[key].origin == "en"
    # ... and German is used when there is no English candidate
    assert extract.select_best([mk("de", .9, 0.1)], wavs, min_conf=.7)[key].origin == "de"
