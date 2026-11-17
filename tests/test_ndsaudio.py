import struct

import numpy as np

from voice2utau import ndsaudio


def _swar(samples):
    """Build a SWAR from [(wave_type, rate, payload bytes)]."""
    n = len(samples)
    head = 16 + 8 + 32 + 4 + 4 * n
    offs, blobs, pos = [], [], head
    for wtype, rate, data in samples:
        data = data + b"\0" * (-len(data) % 4)
        swav = struct.pack("<BBHHHI", wtype, 0, rate, 0, 0, len(data) // 4) + data
        offs.append(pos)
        blobs.append(swav)
        pos += len(swav)
    body = b"DATA" + struct.pack("<I", pos - 16) + b"\0" * 32 + struct.pack("<I", n) + struct.pack(f"<{n}I", *offs)
    return b"SWAR" + struct.pack("<HHIHH", 0xFEFF, 0x0100, pos, 16, 1) + body + b"".join(blobs)


def test_pcm16_and_pcm8_samples_are_read_back():
    pcm = (np.sin(np.arange(400) / 5) * 20000).astype("<i2")
    buf = _swar([(1, 32000, pcm.tobytes()), (0, 16000, bytes([0, 64, 127, 192]))])
    s = ndsaudio.parse_swar(buf)
    assert [r for _, r in s] == [32000, 16000]
    assert np.allclose(s[0][0][:400], pcm / 32768.0, atol=1e-6)
    assert np.allclose(s[1][0][:4], np.array([0, 64, 127, -64]) / 128.0)


def test_ima_adpcm_decoder_steps():
    # predictor 0, step index 40; code 7 (+all bits) then 15 (negative) -> up then back down, index adapts
    data = struct.pack("<hBB", 0, 40, 0) + bytes([0x77, 0xFF])
    y = ndsaudio.decode_ima(data)
    assert len(y) == 4 and y[0] > 0 and y[1] > y[0] and y[3] < y[2]
