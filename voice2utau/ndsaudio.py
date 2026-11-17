"""Extract the raw wave samples from Nintendo DS/DSi sound archives (.swar, or the wave archives inside a .sdat).

A SWAR holds SWAV samples (PCM8, PCM16 or IMA-ADPCM). Inside an SDAT the wave archives are found through the
SYMB (names), INFO (file ids) and FAT (offsets) blocks; a bare SWAR is read directly.

    python -m voice2utau.ndsaudio ARCHIVE [ARCHIVE ...] -o OUT_DIR [--only SUBSTRING]
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_STEP = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97,
         107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
         876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871,
         5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
         27086, 29794, 32767]
_INDEX = [-1, -1, -1, -1, 2, 4, 6, 8]


def decode_ima(data: bytes) -> np.ndarray:
    """DS IMA-ADPCM: 4-byte header (int16 predictor, u8 step index, pad), then 4-bit codes, low nibble first."""
    if len(data) < 4:
        return np.zeros(0, dtype=np.int16)
    pred, idx = struct.unpack_from("<hB", data, 0)
    idx = min(max(idx, 0), 88)
    out = np.empty((len(data) - 4) * 2, dtype=np.int16)
    k = 0
    for byte in data[4:]:
        for code in (byte & 0x0F, byte >> 4):
            step = _STEP[idx]
            diff = step >> 3
            if code & 1:
                diff += step >> 2
            if code & 2:
                diff += step >> 1
            if code & 4:
                diff += step
            pred = max(-32768, pred - diff) if code & 8 else min(32767, pred + diff)
            idx = min(max(idx + _INDEX[code & 7], 0), 88)
            out[k] = pred
            k += 1
    return out


def parse_swav(buf: bytes, off: int, end: int) -> tuple[np.ndarray, int] | None:
    """One SWAV sample at `off` (12-byte info + data). Returns (float32 mono, sample rate)."""
    if off + 12 > len(buf):
        return None
    wtype, loop, rate, timer, loop_off, nonloop = struct.unpack_from("<BBHHHI", buf, off)
    n_bytes = (loop_off + nonloop) * 4
    data = buf[off + 12: min(end, off + 12 + n_bytes)]
    if rate == 0 or not data:
        return None
    if wtype == 0:
        x = np.frombuffer(data, dtype=np.int8).astype(np.float32) / 128.0
    elif wtype == 1:
        x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif wtype == 2:
        x = decode_ima(data).astype(np.float32) / 32768.0
    else:
        return None
    return x, int(rate)


def parse_swar(buf: bytes, base: int = 0) -> list[tuple[np.ndarray, int]]:
    if buf[base:base + 4] != b"SWAR":
        raise ValueError("not a SWAR")
    size = struct.unpack_from("<I", buf, base + 8)[0]
    data = base + 16                                       # 'DATA' block
    if buf[data:data + 4] != b"DATA":
        raise ValueError("SWAR without DATA block")
    count = struct.unpack_from("<I", buf, data + 8 + 32)[0]
    offs = list(struct.unpack_from(f"<{count}I", buf, data + 8 + 32 + 4))
    ends = offs[1:] + [size]
    out = []
    for o, e in zip(offs, ends):
        s = parse_swav(buf, base + o, base + e)
        if s is not None:
            out.append(s)
    return out


def _cstr(buf: bytes, off: int) -> str:
    end = buf.index(b"\0", off)
    return buf[off:end].decode("ascii", "replace")


def sdat_wave_archives(buf: bytes) -> list[tuple[str, bytes]]:
    """[(name, swar bytes)] for every wave archive in an SDAT (names from SYMB when present)."""
    if buf[:4] != b"SDAT":
        raise ValueError("not an SDAT")
    symb_off, symb_size, info_off, info_size, fat_off, fat_size = struct.unpack_from("<6I", buf, 0x10)
    # FAT
    nfat = struct.unpack_from("<I", buf, fat_off + 8)[0]
    fat = [struct.unpack_from("<II", buf, fat_off + 12 + 16 * i) for i in range(nfat)]
    # INFO: 8 record-list offsets; WAVEARC is the 4th (index 3)
    info_list = struct.unpack_from("<8I", buf, info_off + 8)
    wl = info_off + info_list[3]
    nw = struct.unpack_from("<I", buf, wl)[0]
    entries = []
    for i in range(nw):
        e = struct.unpack_from("<I", buf, wl + 4 + 4 * i)[0]
        if e == 0:
            continue
        file_id = struct.unpack_from("<H", buf, info_off + e)[0]
        entries.append((i, file_id))
    names: dict[int, str] = {}
    if symb_off and symb_size:
        symb_list = struct.unpack_from("<8I", buf, symb_off + 8)
        if symb_list[3]:
            sl = symb_off + symb_list[3]
            ns = struct.unpack_from("<I", buf, sl)[0]
            for i in range(ns):
                p = struct.unpack_from("<I", buf, sl + 4 + 4 * i)[0]
                if p:
                    names[i] = _cstr(buf, symb_off + p)
    out = []
    for i, fid in entries:
        if fid < len(fat):
            o, s = fat[fid]
            out.append((names.get(i, f"WAVEARC_{i}"), buf[o:o + s]))
    return out


def extract(path: Path, out_dir: Path, only: str | None = None, prefix: str = "") -> list[Path]:
    from . import audio
    buf = Path(path).read_bytes()
    archives = sdat_wave_archives(buf) if buf[:4] == b"SDAT" else [(Path(path).stem, buf)]
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, swar in archives:
        if only and only.lower() not in name.lower():
            continue
        try:
            samples = parse_swar(swar)
        except ValueError:
            continue
        for k, (x, sr) in enumerate(samples):
            p = out_dir / f"{prefix}{name}_{k:03d}.wav"
            audio.write_wav(p, np.clip(x, -1, 1).astype("float32"), sr)
            written.append(p)
    return written


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archives", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--only", help="only wave archives whose name contains this (e.g. SE)")
    a = ap.parse_args()
    for p in a.archives:
        w = extract(p, a.out, a.only, prefix=p.stem.split("__")[0].replace(" ", "_") + "_")
        print(f"{p.name}: {len(w)} samples")


if __name__ == "__main__":
    main()
