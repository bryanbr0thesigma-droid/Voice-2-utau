"""End-to-end run on synthetic Japanese speech. Needs the phoneme model (~1.2 GB download) and
espeak-ng, so it only runs with V2U_E2E=1:   V2U_E2E=1 pytest tests/test_e2e.py"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voice2utau import audio, pipeline
from voice2utau.morae import MORAE

pytestmark = pytest.mark.skipif(os.environ.get("V2U_E2E") != "1", reason="set V2U_E2E=1 to run")

SENTENCES = ["こんにちは、きょうはいいてんきですね", "さくらがさいています", "わたしのなまえはみくです",
             "ありがとうございます", "ぼくはがっこうへいきます", "ねこがすきです、いぬもすきです",
             "ひゃくにんのきゃくがきました", "りょこうはたのしいです"]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    d = tmp_path_factory.mktemp("rec")
    parts = []
    for i, t in enumerate(SENTENCES * 2):
        w = d / "s.wav"
        subprocess.run(["espeak-ng", "-v", "ja+f3", "-s", "130", "-w", str(w), t], check=True)
        x, sr = audio.read_wav(w)
        parts += [audio.resample(x, sr, 44100), np.zeros(30000, np.float32)]
    audio.write_wav(d / "long.wav", np.concatenate(parts), 44100)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "long.wav"), str(d / "long.mp3")], check=True)
    return d / "long.mp3"


def test_long_mp3_without_rvc_leaves_gaps_and_exports_dataset(recording, tmp_path):
    res = pipeline.run(recording, tmp_path / "w", tmp_path / "o", pipeline.Options(name="E2E"))
    c = res.report["counts"]
    assert c["recorded"] >= 15 and c["rvc"] == 0 and c["missing"] == len(MORAE) - c["recorded"]
    assert res.dataset_zip and res.dataset_zip.exists()
    assert any("no RVC model" in w for w in res.report["warnings"])


def test_with_rvc_command_fills_everything(recording, tmp_path):
    fake = Path(__file__).parent / "fake_rvc.py"
    model = tmp_path / "m.pth"
    model.write_bytes(b"x")
    opt = pipeline.Options(name="E2E", rvc_model=model, rvc_command=f"{sys.executable} {fake} {{input}} {{output}} {{transpose}}")
    res = pipeline.run(recording, tmp_path / "w", tmp_path / "o", opt)
    c = res.report["counts"]
    assert c["missing"] == 0 and c["rvc"] > 0 and c["recorded"] + c["rvc"] == len(MORAE)
    assert (res.bank_dir / "oto.ini").read_bytes().decode("cp932").count("\n") == len(MORAE)


EN_SENTENCES = ["The quick brown fox jumps over the lazy dog", "She sells sea shells by the sea shore",
                "How much wood would a woodchuck chuck", "Peter picked a peck of pickled peppers",
                "Please call Stella and ask her to bring these things", "We think that the weather is rather nice today",
                "A large box of fresh vegetables arrived at noon", "Which witch watched the old Swiss clock"]


@pytest.fixture(scope="module")
def en_recording(tmp_path_factory):
    d = tmp_path_factory.mktemp("en")
    parts = []
    for t in EN_SENTENCES * 2:
        w = d / "s.wav"
        subprocess.run(["espeak-ng", "-v", "en-us+f3", "-s", "150", "-w", str(w), t], check=True)
        x, sr = audio.read_wav(w)
        parts += [audio.resample(x, sr, 44100), np.zeros(30000, np.float32)]
    audio.write_wav(d / "en.wav", np.concatenate(parts), 44100)
    return d / "en.wav"


def test_english_bank_fills_all_675_units(en_recording, tmp_path):
    from voice2utau import english
    fake = Path(__file__).parent / "fake_rvc.py"
    model = tmp_path / "m.pth"
    model.write_bytes(b"x")
    opt = pipeline.Options(name="EN", language="en", rvc_model=model,
                           rvc_command=f"{sys.executable} {fake} {{input}} {{output}} {{transpose}}")
    res = pipeline.run(en_recording, tmp_path / "w", tmp_path / "o", opt)
    c = res.report["counts"]
    assert res.report["language"] == "en" and res.report["settings"]["cross_check"] is False
    assert c["total_target"] == len(english.UNITS) == 675
    assert c["recorded"] >= 20 and c["missing"] == 0 and c["recorded"] + c["rvc"] == 675
    lines = (res.bank_dir / "oto.ini").read_bytes().decode("cp932").splitlines()
    assert len(lines) == 675
    assert any(l.startswith("cv_k_ae.wav=k ae,0,") for l in lines)
    assert any(l.startswith("vc_ae_t.wav=ae t,0,") for l in lines)
    assert "English CVVC" in (res.bank_dir / "readme.txt").read_text()


DE_SENTENCES = ["Guten Morgen, wie geht es dir heute", "Ich möchte gerne ein Glas Wasser trinken",
                "Das Wetter ist heute sehr schön", "Wir gehen zusammen in die Stadt"]


def test_bilingual_zip_converts_german_fallback_through_rvc(tmp_path):
    import zipfile
    from voice2utau import english
    en_dir = tmp_path / "zip"
    (en_dir / "en").mkdir(parents=True)
    (en_dir / "de").mkdir()
    for i, (voice, texts, folder) in enumerate([("en-us+f3", EN_SENTENCES * 2, "en"), ("de+f3", DE_SENTENCES * 3, "de")]):
        for j, t in enumerate(texts):
            subprocess.run(["espeak-ng", "-v", voice, "-s", "150", "-w", str(en_dir / folder / f"{j:02d}.wav"), t], check=True)
    zp = tmp_path / "voice.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for p in sorted(en_dir.rglob("*.wav")):
            z.write(p, p.relative_to(en_dir))
    fake = Path(__file__).parent / "fake_rvc.py"
    model = tmp_path / "m.pth"
    model.write_bytes(b"x")
    opt = pipeline.Options(name="BI", language="en", rvc_model=model,
                           rvc_command=f"{sys.executable} {fake} {{input}} {{output}} {{transpose}}")
    res = pipeline.run(zp, tmp_path / "w", tmp_path / "o", opt)
    r = res.report
    assert {s["lang"] for s in r["sources"]} == {"en", "de"}
    de = [s for s in r["samples"] if s["lang"] == "de"]
    assert de, "German should fill at least one unit the English lines lack"
    assert all(s["source"] == "rvc" for s in de)                      # converted to the main voice
    assert r["counts"]["fallback_converted"] == len(de)
    assert any("converted with your RVC model" in w for w in r["warnings"])
    assert r["counts"]["missing"] == 0 and len(english.UNITS) == 675
