import json

import numpy as np

from voice2utau import audio, crossling
from voice2utau.morae import MORAE

SR = audio.SR_BANK


def make_bank(tmp_path, keys):
    root = tmp_path / "bank"
    root.mkdir()
    lines, samples = [], []
    for i, (k, src) in enumerate(keys.items()):
        x = (0.2 * np.sin(np.arange(int(SR * 0.3)) / (5 + i))).astype(np.float32)
        audio.write_wav(root / f"{k}.wav", x, SR)
        lines.append(f"{k}.wav={k.replace('_', ' ')},0,120,-300,80,26.7")
        samples.append({"key": k, "source": src})
    (root / "oto.ini").write_bytes(("\n".join(lines) + "\n").encode("cp932"))
    (root / "voice2utau_report.json").write_text(json.dumps({"name": "x", "samples": samples}))
    (root / "character.txt").write_bytes(b"name=x\n")
    return root


def test_direct_and_spliced_aliases(tmp_path):
    bank = make_bank(tmp_path, {"cv_k_aa": "recorded", "cv_k_iy": "recorded", "cv_y_aa": "recorded",
                                "cv_t_uw": "recorded", "cv_s_uw": "rvc", "vc_ah_n": "recorded", "v_aa": "recorded"})
    s = crossling.add_japanese(bank, tmp_path / "out", "VLGR")
    root = tmp_path / "out" / "VLGR"
    lines = (root / "oto.ini").read_bytes().decode("cp932").splitlines()
    by_alias = {l.split("=", 1)[1].split(",")[0]: l for l in lines}
    assert by_alias["か"].startswith("cv_k_aa.wav=か,0,") and by_alias["ka"].startswith("cv_k_aa.wav=ka,0,")
    assert by_alias["あ"].startswith("v_aa.wav=")                     # initial-vowel unit
    for a in ("きゃ", "つ", "ん"):                                    # spliced files exist and are referenced WITH .wav
        wav = by_alias[a].split("=")[0]
        assert wav.startswith("ja_") and wav.endswith(".wav") and (root / wav).exists()
    kya = by_alias["きゃ"].split("=")[1].split(",")
    dur = audio.duration(root / "ja_kya.wav") * 1000
    assert 80 < float(kya[4]) < dur              # pre = "k" part + the clip's own consonant
    assert (root / "character.txt").read_bytes().decode("cp932").startswith("name=VLGR")
    assert "か" in s["map"] and s["spliced"] >= 3 and "ぎゃ" in s["missing"]      # no cv_g_iy in this tiny bank


def test_recorded_clips_are_preferred_over_rvc(tmp_path):
    bank = make_bank(tmp_path, {"cv_k_aa": "rvc", "cv_k_ah": "recorded"})
    m = crossling.Mapper(bank)
    assert m.pick(["k"], ["aa", "ah"]) == "cv_k_ah"
    assert m.pick(["k"], ["ae"]) is None


def test_inventory_is_fully_mappable():
    for mora in MORAE.values():
        if mora.key == "nn":
            continue
        if mora.cls:
            assert mora.cls in crossling.CONS or mora.cls in crossling.PALATAL or mora.cls == "ts", mora
        assert not mora.vowel or mora.vowel in crossling.VOWEL
