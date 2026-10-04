import numpy as np
import pytest

from voice2utau import english
from voice2utau.corpus import KEEP, KEEP_FOREIGN, Corpus, CorpusError, file_sha1
from voice2utau.extract import Clip, Option, choose

U = english.UNITS
K1, K2 = "cv_k_ae", "vc_ae_t"


def opt(key, total, f0=200.0, foreign=False, std=0.5, tag=""):
    c = Clip(U[key], (0.1 * np.sin(np.arange(4000) / 9)).astype(np.float32), 0.05, f0, "recorded", 0.9, tag,
             {"f0_std": std, "lang": "de" if foreign else "en"})
    return Option(total, c, foreign)


def test_roundtrip_and_trim(tmp_path):
    c = Corpus(tmp_path, "en", U)
    c.add({K1: [opt(K1, 1.0 + i / 100, tag=f"a{i}") for i in range(KEEP + 5)] + [opt(K1, 2, foreign=True) for _ in range(6)]},
          {"sha1": "x", "name": "a.zip", "files": 3, "seconds": 10.0})
    assert sum(not o.foreign for o in c.options[K1]) == KEEP
    assert sum(o.foreign for o in c.options[K1]) == KEEP_FOREIGN
    c.save()
    d = Corpus.load(tmp_path, "en", U)
    assert len(d.options[K1]) == KEEP + KEEP_FOREIGN and d.stats()["uploads"] == 1
    assert d.options[K1][0].clip.audio.dtype == np.float32 and d.options[K1][0].clip.mora is U[K1]


def test_later_upload_can_replace_earlier_clips(tmp_path):
    c = Corpus(tmp_path, "en", U)
    c.add({K1: [opt(K1, 1.0, tag="old")]}, {"sha1": "1", "files": 1, "seconds": 1})
    c.add({K1: [opt(K1, 1.5, tag="new")], K2: [opt(K2, 1.0, tag="vc")]}, {"sha1": "2", "files": 1, "seconds": 1})
    chosen = choose(c.options)
    assert chosen[K1].origin == "new" and chosen[K2].origin == "vc"      # best across uploads, new unit added
    assert c.has_upload("1") and c.has_upload("2") and not c.has_upload("3")


def test_trim_prefers_steady_pitch_near_median(tmp_path):
    c = Corpus(tmp_path, "en", U)
    # equal base score; wobbly/off-pitch takes should lose when the pool is full
    good = [opt(K1, 1.0, f0=200, std=0.3, tag=f"g{i}") for i in range(KEEP)]
    bad = [opt(K1, 1.0, f0=330, std=5.0, tag=f"b{i}") for i in range(KEEP)]
    c.add({K1: good + bad}, {"sha1": "1", "files": 1, "seconds": 1})
    assert all(o.clip.origin.startswith("g") for o in c.options[K1] if not o.foreign)


def test_language_mismatch_and_bad_format(tmp_path):
    Corpus(tmp_path, "en", U).save()
    with pytest.raises(CorpusError):
        Corpus.load(tmp_path, "ja", {})
    (tmp_path / "state.json").write_text('{"format": 99}')
    with pytest.raises(CorpusError):
        Corpus.load(tmp_path, "en", U)


def test_file_sha1(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    assert file_sha1(tmp_path / "a") == "a9993e364706816aba3e25717850c26c9cd0d89d"
