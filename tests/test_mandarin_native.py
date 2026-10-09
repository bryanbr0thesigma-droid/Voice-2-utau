import shutil
import zipfile

import numpy as np
import pytest

from voice2utau import audio, mandarin_native, rvc

needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")


class CopyBackend(rvc.RVCBackend):
    """Stand-in for RVC: returns the input untouched."""
    name = "copy"

    def convert(self, src, dst, transpose):
        shutil.copy(src, dst)


def _bank(tmp_path):
    b = tmp_path / "bank"
    b.mkdir()
    t = np.arange(int(44100 * 1.2)) / 44100
    audio.write_wav(b / "000.wav", (0.3 * np.sin(2 * np.pi * 170 * t)).astype("float32"), 44100)
    audio.write_wav(b / "sample.wav", (0.3 * np.sin(2 * np.pi * 300 * t)).astype("float32"), 44100)
    (b / "oto.ini").write_bytes("000.wav=er,0,100,-500,50,20\r\n000.wav=あ,0,100,-500,50,20\r\n".encode("cp932"))
    return b


def test_read_aliases_and_pitch(tmp_path):
    b = _bank(tmp_path)
    assert mandarin_native.read_aliases(b) == {"er", "あ"}
    assert 160 < mandarin_native.estimate_f0(b) < 180


def test_build_requires_voice_model_unless_preview(tmp_path):
    with pytest.raises(rvc.RVCError):
        mandarin_native.build(tmp_path, "ZH", 170.0, None)


@needs_espeak
def test_build_merge_and_alias_collision(tmp_path):
    b = _bank(tmp_path)
    root = mandarin_native.build(tmp_path / "out", "ZH", 170.0, CopyBackend(), taken=mandarin_native.read_aliases(b),
                                 only=["ma", "zhuang", "er", "lv"])
    lines = (root / "oto.ini").read_bytes().decode("cp932").splitlines()
    by = {l.split("=", 1)[1].split(",")[0]: l for l in lines}
    assert set(by) == {"ma", "zhuang", "er_zh", "lv"}                 # 'er' is already ARPAbet in the bank
    assert by["ma"].startswith("zh_ma.wav=ma,0,") and (root / "zh_ma.wav").exists()
    pre = float(by["zhuang"].split(",")[4])
    assert 15 <= pre <= audio.duration(root / "zh_zhuang.wav") * 600
    assert mandarin_native.merge_into(b, root) == 4
    assert {"ma", "er_zh", "er", "あ"} <= mandarin_native.read_aliases(b)
    with pytest.raises(ValueError):                                     # second merge would duplicate
        mandarin_native.merge_into(b, root)


def test_export_dataset_skips_sample_and_short_clips(tmp_path):
    b = _bank(tmp_path)
    audio.write_wav(b / "short.wav", np.zeros(4000, dtype="float32"), 44100)
    info = mandarin_native.export_dataset([b], tmp_path / "t.zip")
    assert info["clips"] == 1
    names = zipfile.ZipFile(tmp_path / "t.zip").namelist()
    assert "TRAINING.txt" in names and sum(n.endswith(".wav") for n in names) == 1
