"""Turn an uploaded zip / long audio file into a list of normalised wav sources."""
from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import audio

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".mp4", ".webm", ".mkv"}
MAX_ZIP_UNCOMPRESSED = 8 * 1024 ** 3     # 8 GiB
MAX_ZIP_FILES = 20000


class IngestError(ValueError):
    pass


def safe_extract_zip(zip_path: Path, dest: Path, allowed_ext: set[str] | None = None) -> list[Path]:
    """Extract only regular files with allowed extensions; refuse path traversal and zip bombs."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        raise IngestError("the uploaded file is not a valid zip archive") from e
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_FILES:
            raise IngestError(f"zip has too many entries ({len(infos)})")
        if sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
            raise IngestError("zip is too large when uncompressed")
        for info in infos:
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                continue   # never follow symlinks
            name = info.filename.replace("\\", "/")
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            ext = Path(name).suffix.lower()
            if allowed_ext is not None and ext not in allowed_ext:
                continue
            target = (dest / name).resolve()
            if dest not in target.parents:
                raise IngestError(f"unsafe path in zip: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with zf.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > info.file_size + 1024:
                        raise IngestError(f"zip entry {info.filename!r} is larger than it declares")
                    dst.write(chunk)
            out.append(target)
    return out


SOURCE_LANGS = ("en", "de")
_LANG_WORDS = {"en": {"en", "eng", "english", "enus", "en-us", "en-gb"},
               "de": {"de", "ger", "deu", "german", "deutsch", "de-de"}}


def detect_lang(rel_path: str) -> str | None:
    """Language tag from a zip member path: a folder (`de/line1.wav`) or a file-name token (`line1_de.wav`)."""
    import re
    parts = Path(rel_path.replace("\\", "/")).parts
    tokens = [t.lower() for t in parts[:-1]]
    tokens += re.split(r"[\s_.\-()\[\]]+", Path(parts[-1]).stem.lower())
    for lang, words in _LANG_WORDS.items():
        if any(t in words for t in tokens):
            return lang
    return None


@dataclass
class Source:
    id: str
    original: str
    wav44: Path
    wav16: Path
    seconds: float
    lang: str = "en"          # language spoken in this file (selects which sounds may be used)


def prepare_sources(upload: Path, work: Path, progress=None, default_lang: str = "en") -> list[Source]:
    """Extract (if zip) and decode every audio file to 44.1 kHz and 16 kHz mono wavs."""
    work.mkdir(parents=True, exist_ok=True)
    if upload.suffix.lower() == ".zip":
        files = [p for p in safe_extract_zip(upload, work / "raw", AUDIO_EXT)]
        files.sort()
        raw_root = (work / "raw").resolve()
        if not files:
            raise IngestError("the zip does not contain any audio files "
                              f"(supported: {', '.join(sorted(AUDIO_EXT))})")
    else:
        if upload.suffix.lower() not in AUDIO_EXT:
            raise IngestError(f"unsupported file type {upload.suffix!r}; upload a .zip or an audio file")
        files = [upload]
    if default_lang not in SOURCE_LANGS:
        raise IngestError(f"source language must be one of {SOURCE_LANGS}")
    sources: list[Source] = []
    for i, f in enumerate(files):
        sid = f"{i:04d}"
        w44, w16 = work / f"{sid}_44k.wav", work / f"{sid}_16k.wav"
        try:
            audio.ffmpeg_decode_to_wav(f, w44, audio.SR_BANK)
            audio.ffmpeg_decode_to_wav(f, w16, audio.SR_REC)
        except audio.AudioError:
            if len(files) == 1:
                raise
            continue   # skip undecodable members of a zip
        secs = audio.duration(w44)
        if secs < 0.2:
            continue
        lang = default_lang
        if upload.suffix.lower() == ".zip":
            lang = detect_lang(str(f.resolve().relative_to(raw_root))) or default_lang
        sources.append(Source(sid, f.name, w44, w16, secs, lang))
        if progress:
            progress((i + 1) / len(files))
    if not sources:
        raise IngestError("none of the audio files could be decoded")
    return sources
