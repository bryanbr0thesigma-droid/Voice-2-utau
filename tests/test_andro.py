from pathlib import Path

import numpy as np

from voice2utau import andro, synthbank


def test_voice_names_caught_by_whole_word():
    yes = ["g7_Hello_Motorola_Moto_G7_Power", "oneui5_Bixby_BOS", "aosp_Beat_Box_Android", "galaxys_Vocalise",
           "miui_dynamic_alarm_speech", "droid"]
    no = ["android_low_battery", "s2_Android_keyboard_tone_1", "note5_Crossing_a_River", "oneui5_Single_Tone",
          "mitv_android_TV_select", "aosp_F1_NewVoicemail"]
    assert all(andro.named_like_voice(Path(f"{s}.wav")) for s in yes)
    assert not any(andro.named_like_voice(Path(f"{s}.wav")) for s in no)


def test_male_profile_is_temporary():
    f0, sc = synthbank.F0, synthbank.FORMANT_SCALE
    with andro.male():
        assert synthbank.F0 == andro.MALE_F0 and synthbank.FORMANT_SCALE == andro.MALE_SCALE
    assert (synthbank.F0, synthbank.FORMANT_SCALE) == (f0, sc)


def test_formant_scale_lowers_the_vowel():
    from voice2utau.morae import MORAE
    def f1(scale):
        old = synthbank.FORMANT_SCALE
        synthbank.FORMANT_SCALE = scale
        try:
            y, pre = synthbank.synth(MORAE["a"], np.random.default_rng(0))
        finally:
            synthbank.FORMANT_SCALE = old
        seg = y[int(0.1 * synthbank.SR):int(0.4 * synthbank.SR)]
        X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); f = np.fft.rfftfreq(len(seg), 1 / synthbank.SR)
        band = (f > 300) & (f < 1500)
        return (X[band] * f[band]).sum() / X[band].sum()
    assert f1(0.86) < f1(1.0) * 0.95


def test_merge_set_adds_suffixed_aliases(tmp_path):
    core, part = tmp_path / "core", tmp_path / "part"
    core.mkdir(); part.mkdir()
    (core / "oto.ini").write_bytes("a.wav=あ,0,60,-500,20,7\n".encode("cp932"))
    (part / "a.wav").write_bytes(b"RIFF")
    (part / "oto.ini").write_bytes("a.wav=あ,0,60,-500,20,7\na.wav=a,0,60,-500,20,7\n".encode("cp932"))
    assert andro.merge_set(core, part, "sam") == 1
    lines = (core / "oto.ini").read_bytes().decode("cp932").split()
    assert "a_sam.wav=あ_sam,0,60,-500,20,7" in lines and "a_sam.wav=a_sam,0,60,-500,20,7" in lines
    assert (core / "a_sam.wav").exists() and "a.wav=あ,0,60,-500,20,7" in lines


def test_excluded_paths_read_from_screen(tmp_path):
    t = tmp_path / "x.tsv"
    t.write_text("0.910\tSpeech\tSpeech\t/s/moto/a.wav\n", encoding="utf-8")
    assert andro.read_excluded(t) == {"/s/moto/a.wav"}


def test_stock_sounds_recognised_in_brand_folders(tmp_path):
    a = tmp_path / "and"; a.mkdir()
    for n in ["aosp_Alarm_Beep_01", "aosp_material_Hassium", "aosp_Tick"]:
        (a / f"{n}.wav").write_bytes(b"")
    stock = andro.stock_names(a)
    assert {"alarmbeep01", "hassium"} <= stock and "tick" not in stock          # too short to match safely
    assert andro.is_stock(Path("z2_Alarm_Beep_01_alarms_Sony_Xperia_Z2.wav"), stock)
    assert andro.is_stock(Path("oneui5_Hassium.wav"), stock)
    assert not andro.is_stock(Path("oneui5_Over_the_Horizon.wav"), stock)


def test_spread_takes_files_across_the_list():
    fs = [Path(f"{i:03d}.wav") for i in range(100)]
    got = andro.spread(fs, 5)
    assert len(got) == 5 and got[0] == fs[0] and got[-1] == fs[-1]
    assert andro.spread(fs[:3], 5) == fs[:3]
