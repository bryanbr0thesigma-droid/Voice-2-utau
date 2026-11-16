import numpy as np
import pytest

from voice2utau import audio, sing, synthbank

pw = pytest.importorskip("pyworld")


def _ust(path):
    path.write_bytes("\r\n".join([
        "[#VERSION]", "UST Version1.2", "[#SETTING]", "Tempo=120.00", "Mode2=True",
        "[#0000]", "Length=480", "Lyric=R", "NoteNum=60",
        "[#0001]", "Length=960", "Lyric=か", "NoteNum=67",
        "[#0002]", "Length=960", "Lyric=を", "NoteNum=72", "PBS=-40", "PBW=80",       # を falls back to お
        "[#0003]", "Length=480", "Lyric=っ", "NoteNum=72",                               # treated as a pause
        "[#0004]", "Length=960", "Lyric=ら", "NoteNum=64", "VBR=65,180,35,20,20,0,0",
    ]).encode("cp932"))


def test_parse_and_timing(tmp_path):
    _ust(tmp_path / "t.ust")
    n = sing.parse_ust(tmp_path / "t.ust")
    assert [x.lyric for x in n] == ["R", "か", "を", "っ", "ら"]
    assert n[1].start == pytest.approx(0.5) and n[1].dur == pytest.approx(1.0) and n[2].pbw == [80.0]


def test_render_follows_the_score(tmp_path):
    bank = synthbank.build(tmp_path, "B")
    _ust(tmp_path / "t.ust")
    notes = sing.parse_ust(tmp_path / "t.ust")
    y, info = sing.Engine(bank).render(notes)
    assert info["missing"] == {} and info["notes"] == 3
    f0, t = pw.dio(y.astype("float64"), audio.SR_BANK, frame_period=5.0)
    for n in (notes[1], notes[2], notes[4]):
        seg = f0[int((n.start + 0.3 * n.dur) * 200):int((n.start + 0.6 * n.dur) * 200)]
        seg = seg[seg > 0]
        want = 440 * 2 ** ((n.notenum - 69) / 12)
        assert len(seg) > 10 and abs(1200 * np.log2(np.median(seg) / want)) < 40, n.lyric
    pause = y[int(2.6 * audio.SR_BANK):int(2.9 * audio.SR_BANK)]                            # inside the っ (2.5-3.0 s)
    assert np.sqrt((pause ** 2).mean()) < 0.02
