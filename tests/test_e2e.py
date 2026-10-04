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
