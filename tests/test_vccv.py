from voice2utau import vccv
from voice2utau.sing import Note


def test_parse_reads_vccv_aliases():
    assert vccv.parse("-sO") == (True, ["s", "ow"], False)
    assert vccv.parse("uk-") == (False, ["ah", "k"], True)
    assert vccv.parse("n sh") == (False, ["n", "sh"], False)
    assert vccv.parse("_rO") == (False, ["r", "ow"], False)
    assert vccv.parse("1ng ng") == (False, ["ih", "ng"], False)
    assert vccv.parse("f9") == (False, ["f", "aw"], False)
    assert vccv.parse("vnn") == (False, ["v", "ax", "n"], False)
    assert vccv.parse("??") is None


def test_units_make_arpasing_aliases():
    assert vccv.units(True, ["s", "ow"], False, None) == ["s ow"]
    assert vccv.units(True, ["ay"], False, None) == ["- ay"]
    assert vccv.units(False, ["ah", "k"], True, "ah") == ["ah k", "k -"]
    assert vccv.units(False, ["ow"], False, "s") == ["s ow"]


def test_convert_splits_a_two_unit_note():
    n = Note(lyric="uk-", length=480, notenum=60, tempo=120.0, start=1.0, dur=0.5)
    out, bad = vccv.convert([n])
    assert not bad and [m.lyric for m in out] == ["ah k", "k -"]
    assert sum(m.length for m in out) == 480 and abs(out[1].start - (1.0 + out[0].dur)) < 1e-9
    assert out[0].length > out[1].length or out[0].lyric.split()[-1] not in vccv.ARPA_VOWELS
